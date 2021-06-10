
(* Random Forets Classifier *)

module A = BatArray
module Ht = BatHashtbl
module IntMap = BatMap.Int
module IntSet = BatSet.Int
module L = BatList

open Printf

type features = int IntMap.t
type class_label = int

type sample = features (* X *) *
              class_label (* y *)

type tree = Leaf of class_label
          | Node of tree (* lhs *) *
                    int * int (* (feature, threshold) *) *
                    tree (* rhs *)

type metric = Gini (* default *)
            | Shannon (* TODO *)
            | MCC (* TODO *)

exception Not_singleton

(* FBR: contribute this functionality to batteries *)
let is_singleton s =
  try
    let must_false = ref false in
    IntSet.iter (fun _x ->
        if !must_false then raise Not_singleton;
        must_false := true
      ) s;
    !must_false (* handle empty set case *)
  with Not_singleton -> false

(* tests
   assert (not (is_singleton IntSet.empty));;
   assert (is_singleton (IntSet.singleton 1));;
   assert (not (is_singleton (IntSet.of_list [1;2])));;
*)

(* a feature with non constant value allows to discriminate samples *)
let collect_non_constant_features samples =
  let feat_vals = Ht.create 11 in
  A.iter (fun (features, _class_label) ->
      IntMap.iter (fun feature value ->
          try
            let prev_values = Ht.find feat_vals feature in
            Ht.replace feat_vals feature (IntSet.add value prev_values)
          with Not_found ->
            Ht.add feat_vals feature (IntSet.singleton value)
        ) features
    ) samples;
  Ht.fold (fun feat vals acc ->
      if is_singleton vals then acc
      else (feat, vals) :: acc
    ) feat_vals []

(* split a node *)
let partition_samples feature threshold samples =
  A.partition (fun (features, _class_label) ->
      (* sparse representation --> 0s almost everywhere *)
      let value = IntMap.find_default 0 feature features in
      value <= threshold
    ) samples

(* how many times we see each class label *)
let class_counts samples =
  let ht = Ht.create 11 in
  A.iter (fun (_features, class_label) ->
      let prev_count = Ht.find_default ht class_label 0 in
      Ht.replace ht class_label (prev_count + 1)
    ) samples;
  ht

(* Formula comes from the book:
   "Hands-on machine learning with sklearn ...", A. Geron.
   Same formula in wikipedia. *)
let gini_impurity samples =
  let n = float (A.length samples) in
  let counts = class_counts samples in
  let sum_pi_squares =
    Ht.fold (fun _class_label count acc ->
        let p_i = (float count) /. n in
        (p_i *. p_i) +. acc
      ) counts 0.0 in
  1.0 -. sum_pi_squares

let metric_of = function
  | Gini -> gini_impurity
  | MCC -> failwith "not implemented yet"
  | Shannon -> failwith "not implemented yet"

(* Formula comes from the book:
   "Hands-on machine learning with sklearn ...", A. Geron.
   It must be minimized. *)
let cost_function metric left right =
  let card_left = A.length left in
  let card_right = A.length right in
  let n = float (card_left + card_right) in
  let w_left = (float card_left) /. n in
  let w_right = (float card_right) /. n in
  ((w_left  *. (metric left)) +.
   (w_right *. (metric right)))

let majority_class rng samples =
  let ht = class_counts samples in
  (* find max count *)
  let max_count =
    Ht.fold (fun _class_label count acc ->
        max count acc
      ) ht 0 in
  (* randomly draw from all those with max_count *)
  let majority_classes =
    A.of_list
      (Ht.fold (fun class_label count acc ->
           if count = max_count then class_label :: acc
           else acc
         ) ht []) in
  Utls.array_rand_elt rng majority_classes

let fst5 (a, _, _, (_, _)) = a

let choose_min_cost rng cost_splits =
  let min_cost = L.min (L.map fst5 cost_splits) in
  let candidates =
    A.of_list
      (L.fold (fun acc (cost, feature, value, (left, right)) ->
           if cost = min_cost then
             (cost, feature, value, (left, right)) :: acc
           else acc
         ) [] cost_splits) in
  Utls.array_rand_elt rng candidates

(* maybe this is called the "Classification And Regression Tree" (CART)
   algorithm in the litterature *)
let tree_grow (rng: Random.State.t) (* seeded RNG *)
    (metric: sample array -> float) (* hyper params *)
    (max_features: int)
    (max_samples: int)
    (min_node_size: int)
    (training_set: sample array) (* dataset *) : tree * int array =
  let bootstrap, oob =
    (* First randomization introduced by random forests:
       bootstrap sampling *)
    (* (training_set, training_set) in *)
    Utls.array_bootstrap_sample_OOB rng max_samples training_set in
  let rec loop samples =
    (* min_node_size is a regularization parameter; it also allows to
     * accelerate tree building by early stopping (maybe interesting
     * for very large datasets) *)
    if A.length samples <= min_node_size then
      Leaf (majority_class rng samples)
    else
      (* collect all non constant features *)
      let split_candidates =
        let all_candidates = collect_non_constant_features samples in
        (* randomly keep only N of them:
           Second randomization introduced by random forests:
           random feature sampling. *)
        L.take max_features (L.shuffle ~state:rng all_candidates) in
      (* select the (feature, threshold) pair minimizing cost *)
      let candidate_splits =
        L.fold (fun acc1 (feature, values) ->
            IntSet.fold (fun value acc2 ->
                (feature, value, partition_samples feature value samples)
                :: acc2
              ) values acc1
          ) [] split_candidates in
      let split_costs =
        L.rev_map (fun (feature, value, (left, right)) ->
            let cost = cost_function metric left right in
            (cost, feature, value, (left, right))
          ) candidate_splits in
      (* choose one split minimizing cost *)
      let _cost, feature, threshold, (left, right) =
        choose_min_cost rng split_costs in
      Node (loop left, feature, threshold, loop right)
  in
  (loop bootstrap, oob)

let rand_max_bound = 1073741823 (* 2^30 - 1 *)

let forest_grow
    ncores rng metric ntrees max_features max_samples min_node_size train =
  (* treat the RNG as a seed stream, for reproducibility
     despite potentially out of order parallel run *)
  let seeds =
    A.init ntrees (fun _ -> BatRandom.State.int rng rand_max_bound) in
  let forest = A.init ntrees (fun _ -> (Leaf 0, [||])) in
  let in_count = ref 0 in
  let out_count = ref 0 in
  Parany.run ncores
    ~demux:(fun () ->
        let seed = seeds.(!in_count) in
        incr in_count;
        seed)
    ~work:(fun seed ->
        let rng' = BatRandom.State.make [|seed|] in
        tree_grow rng' metric max_features max_samples min_node_size train)
    ~mux:(fun (tree, oob) ->
        forest.(!out_count) <- (tree, oob);
        incr out_count
      );
  forest

type int_or_float = Int of int (* exact count *)
                  | Float of float (* proportion *)

type forest_oob = (tree * int array) array
type forest = tree array

let drop_OOB (x: forest_oob): forest =
  A.map fst x

let ratio_to_int mini maxi var_name x =
  Utls.bound_between mini maxi (match x with
      | Int i -> i
      | Float f ->
        let () =
          Utls.enforce (0.0 < f && f <= 1.0)
            (sprintf "RFC.train: %s not in ]0.0,1.0]" var_name) in
        BatFloat.round_to_int (f *. (float maxi))
    )

let train (ncores: int)
    (rng: Random.State.t)
    (metric: metric)
    (ntrees: int)
    (max_features: int_or_float)
    (card_features: int)
    (max_samples: int_or_float)
    (min_node_size: int)
    (train: sample array): forest_oob =
  Utls.enforce (1 <= ntrees) "RFC.train: ntrees < 1";
  let metric_f = metric_of metric in
  let max_feats = ratio_to_int 1 card_features "max_features" max_features in
  let n = A.length train in
  let max_samps = ratio_to_int 1 n "max_samples" max_samples in
  let min_node =
    let () =
      Utls.enforce (1 <= min_node_size && min_node_size < n)
        "RFC.train: min_node_size not in [1,n[" in
    min_node_size in
  forest_grow ncores rng metric_f ntrees max_feats max_samps min_node train

let predict =
  failwith "not implemented yet"

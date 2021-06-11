
(* Random Forets Classifier *)

module A = BatArray
module Ht = BatHashtbl
module IntMap = BatMap.Int
module IntSet = BatSet.Int
module L = BatList
module Log = Dolog.Log
module Rand = BatRandom.State

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
let class_count_samples samples =
  let ht = Ht.create 11 in
  A.iter (fun (_features, class_label) ->
      let prev_count = Ht.find_default ht class_label 0 in
      Ht.replace ht class_label (prev_count + 1)
    ) samples;
  ht

let class_count_labels labels =
  let ht = Ht.create 11 in
  A.iter (fun class_label ->
      let prev_count = Ht.find_default ht class_label 0 in
      Ht.replace ht class_label (prev_count + 1)
    ) labels;
  ht

(* Formula comes from the book:
   "Hands-on machine learning with sklearn ...", A. Geron.
   Same formula in wikipedia. *)
let gini_impurity samples =
  let n = float (A.length samples) in
  let counts = class_count_samples samples in
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
  if A.length samples = 0 then
    assert(false)
  else if A.length samples = 1 then
    snd (samples.(0))
  else
    let ht = class_count_samples samples in
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

let choose_min_cost rng = function
  | [] -> assert(false)
  | [x] -> x
  | cost_splits ->
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
  let rec loop (* depth *) samples =
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
      match split_candidates with
      | [] -> (* cannot discriminate samples further *)
        Leaf (majority_class rng samples)
      | _ ->
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
        (* Log.debug "depth: %d feat: %d thresh: %d cost: %f"
         *   depth feature threshold cost; *)
        if A.length left = 0 then
          Leaf (majority_class rng right)
        else if A.length right = 0 then
          Leaf (majority_class rng left)
        else
          (* let depth' = 1 + depth in *)
          Node (loop (* depth' *) left, feature, threshold,
                loop (* depth' *) right)
  in
  (loop (* 0 *) bootstrap, oob)

let rand_max_bound = 1073741823 (* 2^30 - 1 *)

let array_parmap ncores f a init =
  let n = A.length a in
  let res = A.create n init in
  let in_count = ref 0 in
  let out_count = ref 0 in
  Parany.run ncores
    ~demux:(fun () ->
        if !in_count = n then
          raise Parany.End_of_input
        else
          let x = a.(!in_count) in
          incr in_count;
          x)
    ~work:(fun x -> f x)
    ~mux:(fun y ->
        res.(!out_count) <- y;
        incr out_count);
  res

let forest_grow
    ncores rng metric ntrees max_features max_samples min_node_size train =
  (* treat the RNG as a seed stream, for reproducibility
     despite potentially out of order parallel run *)
  let seeds =
    A.init ntrees (fun _ -> Rand.int rng rand_max_bound) in
  array_parmap ncores
    (fun seed ->
       let rng' = Rand.make [|seed|] in
       tree_grow rng' metric max_features max_samples min_node_size train
    )
    seeds (Leaf 0, [||])

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

(* predict for one sample using one tree *)
let tree_predict tree (features, _label) =
  let rec loop = function
    | Leaf label -> label
    | Node (lhs, feature, threshold, rhs) ->
      let value = IntMap.find_default 0 feature features in
      if value <= threshold then
        loop lhs
      else
        loop rhs in
  loop tree

let predict_one ncores rng forest x =
  let pred_labels =
    array_parmap ncores
      (fun (tree, _oob) -> tree_predict tree x) forest 0 in
  let label_counts = class_count_labels pred_labels in
  let ntrees = float (A.length forest) in
  let label_probabilities =
    Ht.fold (fun label count acc ->
        (label, (float count) /. ntrees) :: acc
      ) label_counts [] in
  let p_max = L.max (L.map snd label_probabilities) in
  let candidates =
    A.of_list (
      L.filter (fun (_label, p) -> p = p_max) label_probabilities
    ) in
  Utls.array_rand_elt rng candidates

(* FBR: (predicted_label, label_probability, margin) *)

(* FBR: check when we really need to create a new RNG *)

(* FBR: store the RNG state along with the tree? *)

(* will scale better than predict_one *)
let predict_many rng ncores forest xs =
  array_parmap ncores (predict_one 1 rng forest) xs (0, 0.0)

(* FBR: parallelize this one *)
let predict_OOB forest train =
  let card_OOB =
    A.fold_left (fun acc (_tree, oob) -> acc + (A.length oob)) 0 forest in
  let truth_preds = A.create card_OOB (0, 0) in
  let i = ref 0 in
  A.iter (fun (tree, oob) ->
      let train_OOB = A.map (fun i -> train.(i)) oob in
      let truths = A.map snd train_OOB in
      let preds = A.map (tree_predict tree) train_OOB in
      A.iter2 (fun truth pred ->
          truth_preds.(!i) <- (truth, pred);
          incr i
        ) truths preds
    ) forest;
  truth_preds

(* MCC for particular class of interest *)
let mcc target_class truth_preds =
  let tp_ = ref 0 in
  let tn_ = ref 0 in
  let fp_ = ref 0 in
  let fn_ = ref 0 in
  A.iter (fun (truth, pred) ->
      match truth = target_class, pred = target_class with
      | true , true  -> incr tp_
      | false, false -> incr tn_
      | true , false -> incr fn_
      | false, true  -> incr fp_
    ) truth_preds;
  let tp = !tp_ in
  let tn = !tn_ in
  let fp = !fp_ in
  let fn = !fn_ in
  float ((tp * tn) - (fp * fn)) /.
  sqrt (float ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

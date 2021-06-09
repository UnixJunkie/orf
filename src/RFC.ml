
(* Random Forets Classifier *)

module A = BatArray
module Ht = BatHashtbl
module IntMap = BatMap.Int
module IntSet = BatSet.Int
module L = BatList

type features = int IntMap.t
type class_label = int

type sample = features (* X *) *
              class_label (* y *)

type tree = Leaf of class_label
          | Node of tree (* lhs *) *
                    int * int (* (feature, value) *) *
                    tree (* rhs *)

type metric = Shannon (* TODO *)
            | MCC (* TODO *)
            | Gini (* default *)

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
let collect_non_constant_features (samples: sample array) =
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
let partition_samples (feature: int) (threshold: int) (samples: sample array) =
  A.partition (fun (features, _class_label) ->
      (* sparse representation: almost 0s everywhere *)
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

(* Formula comes from the book:
   "Hands-on machine learning with sklearn ...", A. Geron.
   It must be minimized. *)
let cost_function metric (left, right) =
  let card_left = A.length left in
  let card_right = A.length right in
  let n = float (card_left + card_right) in
  let w_left = (float card_left) /. n in
  let w_right = (float card_right) /. n in
  ((w_left  *. (metric left)) +.
   (w_right *. (metric right)))

let choose_min_cost rng cost_splits =
  let min_cost = L.min (L.map fst cost_splits) in
  let candidates =
    A.of_list
      (L.fold (fun acc (cost, (left, right)) ->
           if cost = min_cost then (left, right) :: acc
           else acc
         ) [] cost_splits) in
  Utls.array_rand_elt rng candidates
  
(* maybe this is called the "Classification And Regression Tree" (CART)
   algorithm in the litterature *)
let grow_one_tree rng (* repro *)
    metric max_features max_samples min_node_size (* hyper params *)
    training_set (* dataset *) =
  let bootstrap, oob =
    (* First randomization introduced by random forests:
       bootstrap sampling *)
    Utls.array_bootstrap_sample_OOB rng max_samples training_set in
  let rec loop (samples: sample array) =
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
           features sampling. *)
        L.take max_features (L.shuffle ~state:rng all_candidates) in
      (* select the (feature, threshold) pair which maximizes metric *)
      let candidate_splits =
        L.fold (fun acc1 (feature, values) ->
            IntSet.fold (fun value acc2 ->
                (partition_samples feature value samples) :: acc2
              ) values acc1
          ) [] split_candidates in
      let split_costs =
        L.rev_map (fun (left, right) ->
            let cost = cost_function metric (left, right) in
            (cost, (left, right))
          ) candidate_splits in
      (* find min cost *)
      
      (* random choose one split which maximixed metric *)
      failwith "not implemented yet"
  in
  (loop bootstrap, oob)

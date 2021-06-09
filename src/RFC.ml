
(* Random Forets Classifier *)

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
let collect_non_constant_features samples =
  let feat_vals = Ht.create 11 in
  L.iter (fun sample ->
      IntMap.iter (fun feature value ->
          try
            let prev_values = Ht.find feat_vals feature in
            Ht.replace feat_vals feature (IntSet.add value prev_values)
          with Not_found ->
            Ht.add feat_vals feature (IntSet.singleton value)
        ) sample
    ) samples;
  Ht.fold (fun feat vals acc ->
      if is_singleton vals then acc
      else (feat, vals) :: acc
    ) feat_vals []

(* let grow_one_tree rng (\* repro *\)
 *     metric max_features max_samples min_node_size (\* hyper params *\)
 *     training_set (\* dataset *\) =
 *   let bootstrap, oob =
 *     Utls.array_bootstrap_sample_OOB rng max_samples training_set in
 *   let rec loop samples =
 *     (\* collect all non constant features *\)
 *     (\* select the (feature, threshold) pair which maximizes the
 *        metric *\)
 *     (\* split on that and recurse *\)
 *     failwith "not implemented yet"
 *   in
 *   (loop bootstrap, oob) *)

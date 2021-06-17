(* Copyright (C) 2021, Francois Berenger

   Tsuda laboratory, Tokyo university,
   5-1-5 Kashiwa-no-ha, Kashiwa-shi, Chiba-ken, 277-8561, Japan. *)

(* Random Forests Regressor *)

module A = BatArray
module Ht = BatHashtbl
module IntMap = BatMap.Int
module IntSet = BatSet.Int
module L = BatList
module Log = Dolog.Log
module Rand = BatRandom.State

type features = int IntMap.t
type dep_var = float

type sample = features (* X *) *
              dep_var (* y *)

type tree = Leaf of dep_var
          | Node of tree (* lhs *) *
                    int * int (* (feature, threshold) *) *
                    tree (* rhs *)

type metric = MSE (* Mean Squared Error *)
            | MAE (* Mean Absolute Error *)
            | MAD (* Median Absolute Deviation *)

let square x =
  x *. x

let average_dep_vars samples =
  let n = float (A.length samples) in
  let sum =
    A.fold (fun acc (_features, value) ->
        acc +. value
      ) 0.0 samples in
  sum /. n

let mean_squared_error samples =
  let n = float (A.length samples) in
  let avg = average_dep_vars samples in
  let sum_squared_errors =
    A.fold (fun acc (_sample, y) ->
        acc +. square (y -. avg)
      ) 0.0 samples in
  sum_squared_errors /. n

let metric_of = function
  | MSE -> mean_squared_error
  | MAE -> failwith "MAE not implemented yet"
  | MAD -> failwith "MAD not implemented yet"

(* maybe this is called the "Classification And Regression Tree" (CART)
   algorithm in the litterature *)
let tree_grow (rng: Random.State.t) (* seeded RNG *)
    (metric: sample array -> float) (* hyper params *)
    (max_features: int)
    (max_samples: int)
    (min_node_size: int)
    (training_set: sample array) (* dataset *) : tree * int array =
  let bootstrap, oob =
    (* First randomization introduced by random forests: bootstrap sampling *)
    Utls.array_bootstrap_sample_OOB rng max_samples training_set in
  let rec loop samples =
    (* min_node_size is a regularization parameter; it also allows to
     * abort tree building (might be interesting for very large datasets) *)
    if A.length samples <= min_node_size then
      Leaf (average_dep_vars samples)
    else
      (* collect all non constant features *)
      let split_candidates =
        let all_candidates = RFC.collect_non_constant_features samples in
        (* randomly keep only N of them:
           Second randomization introduced by random forests
           (random feature sampling). *)
        L.take max_features (L.shuffle ~state:rng all_candidates) in
      match split_candidates with
      | [] -> (* cannot discriminate samples further *)
        Leaf (average_dep_vars samples)
      | _ ->
        (* select the (feature, threshold) pair minimizing cost *)
        let candidate_splits =
          L.fold (fun acc1 (feature, values) ->
              IntSet.fold (fun value acc2 ->
                  (feature, value, RFC.partition_samples feature value samples)
                  :: acc2
                ) values acc1
            ) [] split_candidates in
        let split_costs =
          L.rev_map (fun (feature, value, (left, right)) ->
              let cost = RFC.cost_function metric left right in
              (cost, feature, value, (left, right))
            ) candidate_splits in
        (* choose one split minimizing cost *)
        let cost, feature, threshold, (left, right) =
          RFC.choose_min_cost rng split_costs in
        if A.length left = 0 then
          Leaf (average_dep_vars right)
        else if A.length right = 0 then
          Leaf (average_dep_vars left)
        else if cost = 0.0 then
          (* if the cost is minimal: pure nodes -> stop digging *)
          Node (Leaf (average_dep_vars left), feature, threshold,
                Leaf (average_dep_vars right))
        else
          Node (loop left, feature, threshold, loop right)
  in
  (loop (* 0 *) bootstrap, oob)

(* array of all samples whose index is listed *)
let extract indexes (samples: sample array): sample array =
  A.map (A.unsafe_get samples) indexes

let rand_max_bound = 1073741823 (* 2^30 - 1 *)

let forest_grow
    ncores rng metric ntrees max_features max_samples min_node_size train =
  (* treat the RNG as a seed stream, for reproducibility
     despite potentially out of order parallel run *)
  let seeds =
    A.init ntrees (fun _ -> Rand.int rng rand_max_bound) in
  RFC.array_parmap ncores
    (fun seed ->
       let rng' = Rand.make [|seed|] in
       tree_grow rng' metric max_features max_samples min_node_size train
    )
    seeds (Leaf 0.0, [||])

type forest = (tree * int array) array

(* before saving a model, we might want to just get rid of the OOB
 * sample indexes *)
let drop_OOB (f: forest): forest =
  A.map (fun (t, _oob) -> (t, [||])) f

let train (ncores: int)
    (rng: Random.State.t)
    (metric: metric)
    (ntrees: int)
    (max_features: RFC.int_or_float)
    (card_features: int)
    (max_samples: RFC.int_or_float)
    (min_node_size: int)
    (train: sample array): forest =
  Utls.enforce (1 <= ntrees) "RFC.train: ntrees < 1";
  let metric_f = metric_of metric in
  let max_feats =
    RFC.ratio_to_int 1 card_features "max_features" max_features in
  let n = A.length train in
  let max_samps =
    RFC.ratio_to_int 1 n "max_samples" max_samples in
  let min_node =
    let () =
      Utls.enforce (1 <= min_node_size && min_node_size < n)
        "RFC.train: min_node_size not in [1,n[" in
    min_node_size in
  forest_grow
    ncores rng metric_f ntrees max_feats max_samps min_node train

(* (\* predict for one sample using one tree *\)
 * let tree_predict tree (features, _label) =
 *   let rec loop = function
 *     | Leaf label -> label
 *     | Node (lhs, feature, threshold, rhs) ->
 *       let value = IntMap.find_default 0 feature features in
 *       if value <= threshold then
 *         loop lhs
 *       else
 *         loop rhs in
 *   loop tree
 * 
 * (\* label to predicted probability hash table *\)
 * let predict_one_proba ncores forest x =
 *   let pred_labels =
 *     array_parmap ncores
 *       (fun (tree, _oob) -> tree_predict tree x) forest 0 in
 *   let label_counts = class_count_labels pred_labels in
 *   let ntrees = float (A.length forest) in
 *   Ht.fold (fun label count acc ->
 *       (label, (float count) /. ntrees) :: acc
 *     ) label_counts []
 * 
 * let predict_one ncores rng forest x =
 *   let label_probabilities = predict_one_proba ncores forest x in
 *   let p_max = L.max (L.rev_map snd label_probabilities) in
 *   let candidates =
 *     A.of_list (
 *       L.filter (fun (_label, p) -> p = p_max) label_probabilities
 *     ) in
 *   Utls.array_rand_elt rng candidates
 * 
 * let predict_one_margin ncores rng forest x =
 *   let label_probabilities = predict_one_proba ncores forest x in
 *   let p_max = L.max (L.rev_map snd label_probabilities) in
 *   let candidates =
 *     A.of_list (
 *       L.filter (fun (_label, p) -> p = p_max) label_probabilities
 *     ) in
 *   let pred_label, pred_proba = Utls.array_rand_elt rng candidates in
 *   let other_label_p_max =
 *     A.fold_left (fun acc (label, p) ->
 *         if label <> pred_label then
 *           max acc p
 *         else
 *           acc
 *       ) 0.0 candidates in
 *   let margin = pred_proba -. other_label_p_max in
 *   (pred_label, pred_proba, margin) *)

(* FBR: (predicted_label, label_probability, margin) *)

(* FBR: check when we really need to create a new RNG *)

(* FBR: store the RNG state along with the tree? *)

(* (\* will scale better than predict_one *\)
 * let predict_many ncores rng forest xs =
 *   array_parmap ncores (predict_one 1 rng forest) xs (0, 0.0)
 * 
 * let predict_many_margin ncores rng forest xs =
 *   array_parmap ncores (predict_one_margin 1 rng forest) xs (0, 0.0, 0.0)
 * 
 * let predict_OOB forest train =
 *   let card_OOB =
 *     A.fold_left (fun acc (_tree, oob) -> acc + (A.length oob)) 0 forest in
 *   let truth_preds = A.create card_OOB (0, 0) in
 *   let i = ref 0 in
 *   A.iter (fun (tree, oob) ->
 *       let train_OOB = extract oob train in
 *       let truths = A.map snd train_OOB in
 *       let preds = A.map (tree_predict tree) train_OOB in
 *       A.iter2 (fun truth pred ->
 *           truth_preds.(!i) <- (truth, pred);
 *           incr i
 *         ) truths preds
 *     ) forest;
 *   truth_preds
 * 
 * (\* MCC for particular class of interest *\)
 * let mcc target_class truth_preds =
 *   let tp_ = ref 0 in
 *   let tn_ = ref 0 in
 *   let fp_ = ref 0 in
 *   let fn_ = ref 0 in
 *   A.iter (fun (truth, pred) ->
 *       match truth = target_class, pred = target_class with
 *       | true , true  -> incr tp_
 *       | false, false -> incr tn_
 *       | true , false -> incr fn_
 *       | false, true  -> incr fp_
 *     ) truth_preds;
 *   let tp = !tp_ in
 *   let tn = !tn_ in
 *   let fp = !fp_ in
 *   let fn = !fn_ in
 *   Log.info "TP: %d TN: %d FP: %d FN: %d" tp tn fp fn;
 *   float ((tp * tn) - (fp * fn)) /.
 *   sqrt (float ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
 * 
 * let accuracy truth_preds =
 *   let n = A.length truth_preds in
 *   let correct_preds = ref 0 in
 *   A.iter (fun (truth, pred) ->
 *       if truth = pred then incr correct_preds
 *     ) truth_preds;
 *   (float !correct_preds) /. (float n)
 * 
 * module Score_label = struct
 *   type t = float * bool
 *   let get_score (s, _l) = s
 *   let get_label (_s, l) = l
 * end
 * 
 * module ROC = Cpm.MakeROC.Make(Score_label)
 * 
 * let roc_auc target_class preds true_labels =
 *   let score_labels =
 *     A.map2 (fun (pred_label, pred_proba) true_label ->
 *         if pred_label = target_class then
 *           (pred_proba, true_label = target_class)
 *         else
 *           (1.0 -. pred_proba, true_label = target_class)
 *       ) preds true_labels in
 *   ROC.auc_a score_labels *)

type filename = string

let save fn forest =
  Utls.save fn forest

let restore fn =
  Utls.restore fn

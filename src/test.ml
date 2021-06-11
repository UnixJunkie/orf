
module A = BatArray
module CLI = Minicli.CLI
module IntMap = BatMap.Int
module L = BatList
module LO = Line_oriented
module Log = Dolog.Log
module RFC = Orf.RFC
module S = BatString

open Printf

let label_of_string = int_of_string

let features_of_str_tokens toks =
  L.fold_left (fun acc tok_str ->
      Scanf.sscanf tok_str "%d:%d" (fun k v ->
          IntMap.add k v acc
        )
    ) IntMap.empty toks

let sample_of_csv_line l =
  let tokens = S.split_on_char ' ' l in
  match tokens with
  | label_str :: feature_values ->
    let label = label_of_string label_str in
    let features = features_of_str_tokens feature_values in
    (features, label)
  | [] -> assert(false)

let load_csv_file fn =
  A.of_list (LO.map fn sample_of_csv_line)

let main () =
  Log.color_on ();
  Log.(set_log_level DEBUG);
  let argc, args = CLI.init () in
  if argc = 0 then
    begin
      printf "%s -tr train.csv \
              -te test.csv \
              -n <ntrees:int> \
              -np <ncores:int>" Sys.argv.(0);
      exit 0
    end;
  let train_set =
    let train_set_fn = CLI.get_string ["-tr"] args in
    load_csv_file train_set_fn in
  let test_set =
    let test_set_fn = CLI.get_string ["-te"] args in
    load_csv_file test_set_fn in
  let ncores = CLI.get_int_def ["-np"] args 1 in
  let ntrees = CLI.get_int_def ["-n"] args 100 in
  let feats_portion = CLI.get_float_def ["-feats"] args 1.0 in
  let samples_portion = CLI.get_float_def ["-samps"] args 1.0 in
  let min_node_size = CLI.get_int_def ["-min"] args 1 in
  CLI.finalize();
  let rng = BatRandom.State.make [|3141593|] in
  let model =
    RFC.(train
           ncores
           rng
           Gini
           ntrees
           (Float feats_portion)
           17368
           (Float samples_portion)
           min_node_size
           train_set) in
  let oob_preds = RFC.predict_OOB model train_set in
  let oob_mcc = RFC.mcc 1 oob_preds in
  let preds = RFC.predict_many rng ncores model test_set in
  let test_true_labels = A.map snd test_set in
  let test_pred_labels = A.map fst preds in
  let test_preds =
    A.map2 (fun x y -> (x, y)) test_true_labels test_pred_labels in
  let test_mcc = RFC.mcc 1 test_preds in
  Log.info "OOB MCC: %f test MCC: %f" oob_mcc test_mcc

let () = main ()
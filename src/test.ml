
module A = BatArray
module IntMap = BatMap.Int
module L = BatList
module LO = Line_oriented
module Log = Dolog.Log
module RFC = Orf.RFC
module S = BatString

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
  let train_set = load_csv_file "data/train.csv" in
  let _test_set = load_csv_file "data/test.csv" in
  let ncores = 1 in
  let ntrees = 1 in
  let model =
    RFC.(train
           ncores
           (BatRandom.State.make [|3141593|])
           Gini
           ntrees
           (Float 0.1)
           17368
           (Float 1.0)
           1
           train_set) in
  let oob_preds = RFC.predict_OOB model train_set in
  let mcc = RFC.mcc 1 oob_preds in
  Log.info "OOB MCC: %f" mcc

let () = main ()

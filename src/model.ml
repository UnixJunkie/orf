(* Copyright (C) 2021, Francois Berenger

   Tsuda laboratory, Tokyo university,
   5-1-5 Kashiwa-no-ha, Kashiwa-shi, Chiba-ken, 277-8561, Japan. *)

open Printf

module A = BatArray
module Buff = Buffer
module CLI = Minicli.CLI
module Feat_vect = Orf.Feature_vector
module Fn = Filename
module Fp = Molenc.Fingerprint
module IntMap = BatMap.Int
module L = BatList
module LO = Line_oriented
module Log = Dolog.Log
module Mol = Molenc.FpMol
module RFR = Orf.RFR
module Stats = Cpm.RegrStats

type model_file_mode = Save of string
                     | Load of string
                     | Save_to_temp

(* FBR: I want R2 and RMSE of the model *)

(* FBR: I also want the actual Vs. pred curve *)

let train_test_NxCV nprocs train_fun test_fun nfolds training_set =
  let folds = Cpm.Utls.cv_folds nfolds training_set in
  Parany.Parmap.parfold nprocs
    (fun (train, test) ->
       let actual = L.map Mol.get_value test in
       let model_fn = Fn.temp_file "Orf_" ".model" in
       train_fun model_fn train;
       let preds_stdevs = test_fun model_fn test in
       L.map2 (fun actual (pred, pred_std) ->
           (actual, pred, pred_std)
         ) actual preds_stdevs
    )
    (fun acc x -> L.rev_append x acc)
    [] folds

(* Orf.RFR needs a samples array *)
let samples_array_of_mols_list mols =
  let n = L.length mols in
  let dummy = (Feat_vect.zero (), 0.0) in
  if n = 0 then
    let () = Log.warn "Model.samples_array_of_mols_list: no mols" in
    A.make 0 dummy
  else
    let res = A.make n dummy in
    L.iteri (fun i mol ->
        let activity = Mol.get_value mol in
        let vec = Feat_vect.zero () in
        let key_values = Fp.key_values (Mol.get_fp mol) in
        IntMap.iter (fun k v ->
            Feat_vect.set k v vec
          ) key_values;
        res.(i) <- (vec, activity)
      ) mols;
    res

let fst3 (a, _, _) = a
let snd3 (_, b, _) = b
let trd3 (_, _, c) = c

let main () =
  Log.color_on ();
  Log.set_log_level Log.INFO;
  Log.info "start";
  let argc, args = CLI.init () in
  let show_help = CLI.get_set_bool ["-h";"--help"] args in
  let train_portion_def = 0.8 in
  let mode = ref Save_to_temp in
  if argc = 1 || show_help then
    begin
      eprintf "usage:\n\
               %s  \
               [-p <float>]: proportion of the (randomized) dataset\n  \
               used to train (default=%.2f)\n  \
               [-np <int>]: max number of processes (default=1)\n  \
               [-n <int>]: |RF|; default=100\n  \
               [-o <filename>]: output scores to file\n  \
               [--train <train.txt>]: training set (overrides -p)\n  \
               [--valid <valid.txt>]: validation set (overrides -p)\n  \
               [--test <test.txt>]: test set (overrides -p)\n  \
               [--NxCV <int>]: number of folds of cross validation\n  \
               [--seed <int>: fix random seed]\n  \
               [--no-regr-plot]: turn OFF regression plot\n  \
               [--rec-plot]: turn ON REC curve\n  \
               [--y-rand]: turn ON Y-randomization\n  \
               [-s <filename>]: save model to file\n  \
               [-l <filename>]: load model from file\n  \
               [--max-feat <int>]: max feature id.  \
               (cf. end of encoding dict)\n  \
               [-v]: verbose/debug mode\n  \
               [-h|--help]: show this help message\n"
        Sys.argv.(0) train_portion_def;
      exit 1
    end;
  let train_portion = ref (CLI.get_float_def ["-p"] args train_portion_def) in
  let nb_trees = CLI.get_int_def ["-n"] args 100 in
  let nprocs = CLI.get_int_def ["-np"] args 1 in
  let maybe_output_fn = CLI.get_string_opt ["-o"] args in
  if CLI.get_set_bool ["--valid"] args then failwith "not implemented yet";
  if CLI.get_set_bool ["--test"] args then failwith "not implemented yet";
  let maybe_nfolds = CLI.get_int_opt ["--NxCV"] args in
  let maybe_seed = CLI.get_int_opt ["--seed"] args in
  let max_feat = CLI.get_int ["--max-feat"] args in
  let no_reg_plot = CLI.get_set_bool ["--no-regr-plot"] args in
  let rec_plot = CLI.get_set_bool ["--rec-plot"] args in
  let _verbose = CLI.get_set_bool ["-v"] args in
  let train_fn = CLI.get_string ["--train"] args in
  let feats_portion = CLI.get_float_def ["--feat"] args 1.0 in
  assert(feats_portion > 0.0 && feats_portion <= 1.0);
  let samples_portion = CLI.get_float_def ["--samples"] args 1.0 in
  assert(samples_portion > 0.0 && samples_portion <= 1.0);
  let min_node_size = CLI.get_int_def ["--min-node"] args 1 in
  begin match CLI.get_string_opt ["-s"] args with
    | None -> ()
    | Some fn ->
      (if !train_portion < 1.0 then Log.warn "p forced to 1.0 because of -s";
       train_portion := 1.0;
       mode := Save fn)
  end;
  begin match CLI.get_string_opt ["-l"] args with
    | None -> ()
    | Some fn ->
      (if !train_portion > 0.0 then Log.warn "p forced to 0.0 because of -l";
       train_portion := 0.0;
       mode := (Load fn))
  end;
  CLI.finalize(); (* ------------------------------------------------------- *)
  let data_dir = Fn.dirname train_fn in
  let rng = match maybe_seed with
    | None -> Random.State.make_self_init ()
    | Some seed -> Random.State.make [|seed|] in
  let model_fn = match !mode with
    | Save_to_temp -> Fn.temp_file "Orf_" ".model"
    | Load fn -> fn
    | Save fn -> fn in
  let nb_features, train_set, test_set =
    let all_lines =
      let ordered = Mol.molecules_of_file train_fn in
      match !mode with
      | Save_to_temp | Save _ ->
        (* destroy any molecule ordering the input file might have *)
        BatList.shuffle ~state:rng ordered
      | Load _fn ->
        (* don't reorder lines in production *)
        ordered in
    let training, testing =
      Cpm.Utls.train_test_split !train_portion all_lines in
    (max_feat + 1,
     samples_array_of_mols_list training,
     samples_array_of_mols_list testing) in
  Log.info "nb_features: %d" nb_features;
  let acts_preds_stdevs = match maybe_nfolds with
    | Some _nfolds ->
      failwith "not implemented yet"
      (* train_test_NxCV nprocs train_fun test_fun nfolds train_set *)
    | None ->
      begin
        begin match !mode with
          | Load _ -> () (* no need to train new model *)
          | Save fn ->
            begin
              let rfr = RFR.(train nprocs rng MSE nb_trees (Float feats_portion)
                               nb_features (Float samples_portion)
                               min_node_size train_set) in
              RFR.save fn rfr;
              exit 0
            end
          | Save_to_temp ->
            begin
              let rfr = RFR.(train nprocs rng MSE nb_trees (Float feats_portion)
                               nb_features (Float samples_portion)
                               min_node_size train_set) in
              RFR.save model_fn rfr;
              exit 0
            end
        end;
        let actual = A.map snd test_set in
        let rfr = RFR.restore model_fn in
        let preds_stdevs = RFR.predict_many 1 rfr test_set in
        A.map2 (fun act (pred, stddev) -> (act, pred, stddev)) actual preds_stdevs
      end in
  (match maybe_output_fn with
   | None -> ()
   | Some fn ->
     LO.with_out_file fn (fun out ->
         A.iter (fun (_act, pred, _stddev) ->
             fprintf out "%f\n" pred
           ) acts_preds_stdevs
       )
  );
  let nfolds = BatOption.default 1 maybe_nfolds in
  let actual = A.map fst3 acts_preds_stdevs in
  let preds = A.map snd3 acts_preds_stdevs in
  let stdevs = A.map trd3 acts_preds_stdevs in
  let r2 = Stats.r2 (A.to_list actual) (A.to_list preds) in
  let plot_title =
    sprintf "T=%s |RF|=%d k=%d R2=%.2f" data_dir nb_trees nfolds r2 in
  Log.info "%s" plot_title;
  if rec_plot then
    Gnuplot.rec_curve (A.to_list actual) (A.to_list preds);
  if not no_reg_plot then
    Gnuplot.regr_plot plot_title (A.to_list actual) (A.to_list preds) (A.to_list stdevs)

let () = main ()

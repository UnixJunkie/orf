open Printf

module Buff = Buffer
module CLI = Minicli.CLI
module Fn = Filename
module Fp = Molenc.Fingerprint
module L = BatList
module Log = Dolog.Log
module Stats = Cpm.RegrStats
module Mol = Molenc.FpMol

let max_feat_id molecules =
  L.fold_left (fun acc mol ->
      BatInt.max acc (Fp.max_feat_id (Mol.get_fp mol))
    ) (-1) molecules

type mode = Save of string
          | Load of string
          | Save_to_temp

let train_model
    nb_features verbose nprocs nb_trees training_set model_fn =
  let train_csv_fn = Fn.temp_file "RFR_train_" ".csv" in
  Common.csv_dump train_csv_fn nb_features training_set;
  let could_train =
    Oranger.RF.train
      ~debug:verbose ~nprocs Regression nb_trees train_csv_fn
      "Y" (* name of the target variable column *)
      model_fn in
  Utls.enforce could_train (fun () ->
      "RFR.train_test: could not train: train_csv_fn: " ^ train_csv_fn)

let test_model
    nb_features verbose nb_trees test_set model_fn maybe_output_fn =
  let test_csv_fn = Fn.temp_file "RFR_test_" ".csv" in
  Common.csv_dump test_csv_fn nb_features test_set;
  let preds_stdevs =
    BatOption.get
      (Oranger.RF.predict ~debug:verbose nb_trees test_csv_fn model_fn) in
  let names_preds_stdevs =
    let names = L.map Mol.get_name test_set in
    L.map2 Utls.prepend2 names preds_stdevs in
  begin match maybe_output_fn with
    | None -> ()
    | Some fn ->
      Utls.with_out_file fn (fun out ->
          L.iter (fun (name, pred, stdev) ->
              fprintf out "%s %f %f\n" name pred stdev
            ) names_preds_stdevs
        )
  end;
  names_preds_stdevs

let train_test_NxCV verbose nprocs nb_trees nb_features nfolds training_set =
  let folds = L.cv_folds nfolds training_set in
  Parany.Parmap.parfold nprocs
    (fun (train, test) ->
       let actual = L.map Mol.get_value test in
       let model_fn = Fn.temp_file "RFR_" ".model" in
       train_model
         nb_features verbose 1 nb_trees train model_fn;
       let names_preds_stdevs =
         test_model nb_features verbose nb_trees test model_fn None in
       L.map2 Utls.prepend3 actual names_preds_stdevs)
    (fun acc x -> L.rev_append x acc)
    [] folds

let y_randomize rng training_set =
  (* since we randomize Y, we don't care about preserving the order
   * of the list *)
  let activities = L.rev_map Mol.get_value training_set in
  let rand_acts = L.shuffle ~state:rng activities in
  L.rev_map2 (fun mol act ->
      Mol.set_value mol act
    ) training_set rand_acts

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
  let verbose = CLI.get_set_bool ["-v"] args in
  let train_fn = CLI.get_string ["--train"] args in
  begin match CLI.get_string_opt ["-s"] args with
    | None -> ()
    | Some fn ->
      (if !train_portion < 1.0 then Log.warn "p forced to 1.0 because of -s";
       train_portion := 1.0;
       mode := (Save fn))
  end;
  begin match CLI.get_string_opt ["-l"] args with
    | None -> ()
    | Some fn ->
      (if !train_portion > 0.0 then Log.warn "p forced to 0.0 because of -l";
       train_portion := 0.0;
       mode := (Load fn))
  end;
  let y_rand = CLI.get_set_bool ["--y-rand"] args in
  CLI.finalize(); (* ------------------------------------------------------- *)
  let data_dir = Fn.dirname train_fn in
  let rng = match maybe_seed with
    | None -> Random.State.make_self_init ()
    | Some seed -> Random.State.make [|seed|] in
  let model_fn = match !mode with
    | Save_to_temp -> Fn.temp_file "RFR_" ".model"
    | Load fn -> fn
    | Save fn -> fn in
  let nb_features, train, test =
    let all_molecules =
      let ordered = Utls.map_on_lines_of_file train_fn Mol.of_string in
      let disordered = match !mode with
        | Save_to_temp | Save _ ->
          (* destroy any molecule ordering the input file might have *)
          BatList.shuffle ~state:rng ordered
        | Load _fn ->
          (* don't reorder molecules in production *)
          ordered in
      if y_rand then
        (* Y-randomization:
           destroy link between structure and response variable *)
        y_randomize rng disordered
      else disordered in
    let training, testing =
      Common.train_test_split !train_portion all_molecules in
    (max_feat + 1, training, testing) in
  Log.info "nb_features: %d" nb_features;
  let acts_names_preds_stdevs = match maybe_nfolds with
    | Some nfolds ->
      train_test_NxCV verbose nprocs nb_trees nb_features nfolds train
    | None ->
      begin
        begin match !mode with
          | Load _ -> () (* no need to train new model *)
          | Save _fn ->
            (train_model nb_features verbose nprocs nb_trees train model_fn;
             exit 0)
          | Save_to_temp ->
            train_model nb_features verbose nprocs nb_trees train model_fn
        end;
        let actual = L.map Mol.get_value test in
        let names_preds_stdevs =
          test_model nb_features verbose nb_trees test model_fn maybe_output_fn in
        L.map2 Utls.prepend3 actual names_preds_stdevs
      end in
  let nfolds = BatOption.default 1 maybe_nfolds in
  let actual = L.map Utls.fst4 acts_names_preds_stdevs in
  let preds = L.map Utls.trd4 acts_names_preds_stdevs in
  let stdevs = L.map Utls.frt4 acts_names_preds_stdevs in
  let r2 = Stats.r2 actual preds in
  let plot_title =
    sprintf "T=%s |RF|=%d k=%d R2=%.2f" data_dir nb_trees nfolds r2 in
  Log.info "%s" plot_title;
  if rec_plot then
    Gnuplot.rec_curve actual preds;
  if not no_reg_plot then
    Gnuplot.regr_plot plot_title actual preds stdevs

let () = main ()

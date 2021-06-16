
module IntMap = BatMap.Int

type features = int IntMap.t
type class_label = int

type sample = features (* X *) *
              class_label (* y *)

type metric = Gini (* default *)
            | Shannon (* TODO *)
            | MCC (* TODO *)

(** trained Random Forests model *)
type forest

type int_or_float = Int of int (* exact count *)
                  | Float of float (* proportion *)

(** [train ncores rng indexing metric ntrees max_features card_features
           max_samples min_node_size training_set] *)
val train: int -> Random.State.t -> bool -> metric -> int ->
  int_or_float -> int -> int_or_float -> int -> sample array -> forest

(** [(pred_label, pred_proba) =
      predict_one ncores rng trained_forest sample] *)
val predict_one: int -> Random.State.t -> forest -> sample
  -> (class_label * float)

(** [(pred_label, pred_proba, pred_margin) =
      predict_one_margin ncores rng trained_forest sample] *)
val predict_one_margin: int -> Random.State.t -> forest -> sample
  -> (class_label * float * float)

(** like [predict_one] but for an array of samples *)
val predict_many: int -> Random.State.t -> forest -> sample array ->
  (class_label * float) array

(** like [predict_one_margin] but for an array of samples *)
val predict_many_margin: int -> Random.State.t -> forest -> sample array ->
  (class_label * float * float) array

(** use a trained forest to predict on the Out Of Bag (OOB) training set
    of each tree. The training_set must be provided in the same order
    than when the model was trained.
    Can be used to get a reliable model performance estimate,
    even if you don't have a left out test set.
    [truth_preds = predict_OOB forest training_set] *)
val predict_OOB: forest -> sample array -> (class_label * class_label) array

(** Matthews Correlation Coefficient (MCC).
    [mcc target_class_label truth_preds] *)
val mcc: class_label -> (class_label * class_label) array -> float

(** Percentage of correct prediction
    [accuracy truth_preds] *)
val accuracy: (class_label * class_label) array -> float

(** ROC AUC
    [roc_auc target_class_label preds true_labels] *)
val roc_auc: class_label -> (class_label * float) array ->
  class_label array -> float

(** make trained model forget OOB samples (reduce model size) *)
val drop_OOB: forest -> forest

type filename = string

(** Save model to file (Marshal) *)
val save: filename -> forest -> unit

(** Restore model from file (Marshal) *)
val restore: filename -> forest

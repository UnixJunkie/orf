
(** A [feature] is accessed via its (integer) index. *)
type feature = int

(** ['a t] is the type of a feature vector with feature
    indexes of type [int] and values of type ['a]. *)
type 'a t

(** Iterate on a feature vector. *)
val iter: (feature -> 'a -> unit) -> 'a t -> unit

(** The zero feature vector. *)
val zero: unit -> 'a t

(** [get feat vec] returns the value associated to [feat]
    in the integer-valued vector [vec]. *)
val get: feature -> int t -> int

(** [set feat val vec] sets the feature at index [feat] to value [val]
    in vector [vec]. *)
val set: feature -> 'a -> 'a t -> unit

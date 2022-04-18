(** A [feature] is encoded as an integer. *)
type feature = int

(** ['a t] is the type of feature vectors with features of type [int]
    and coefficients of type ['a]. *)
type 'a t

(** Iterate on a feature vector. *)
val iter : (feature -> 'a -> unit) -> 'a t -> unit

(** The zero feature vector. *)
val zero : 'a t

(** [get feature vec] returns the coefficient associated to [feature]
    in the integer-valued vector [vec]. *)
val get : feature -> int t -> int

(** [set feature coeff vec] sets the value of [vec] for [feature] to [coeff]. *)
val set : feature -> 'a -> 'a t -> 'a t


type feature = int

type 'a t = (feature, 'a) Hashtbl.t

let iter f vec =
  Hashtbl.iter f vec

let zero () =
  Hashtbl.create 11

let get feat vec =
  try Hashtbl.find vec feat
  with Not_found -> 0

let set ft coef vec =
  Hashtbl.replace vec ft coef

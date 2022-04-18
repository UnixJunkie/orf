type feature = int

type 'a t = (feature, 'a) Hashtbl.t

let iter f vec = Hashtbl.iter f vec

let zero () = Hashtbl.create 11

let get ft vec = try Hashtbl.find vec ft with Not_found -> 0

let set ft coeff vec = Hashtbl.replace vec ft coeff

type feature = int

module IntMap = BatMap.Int

type 'a t = 'a IntMap.t

let iter = IntMap.iter

let zero = IntMap.empty

let get f vec = IntMap.find_default 0 f vec

let set = IntMap.add

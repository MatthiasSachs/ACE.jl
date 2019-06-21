

function eval_basis end
function eval_basis! end
function eval_grad end
function eval_basis_d! end

eval_basis(B, args...) =
   eval_basis!(alloc_B(B), B, args...)

function eval_basis_d(B, args...)
   b = alloc_B(B)
   db = alloc_dB(B)
   store = alloc_temp_d(B, args[end])
   eval_basis_d!(b, db, args..., store)
end

function alloc_B end
function alloc_dB end

function alloc_temp end
function alloc_temp_d end

function transform end
function transform_d end
function fcut end
function fcut_d end

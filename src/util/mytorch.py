import IPython
import torch

# reshapes the tensor by merging the specified dim with the following one
def mergeDim(tensor,dim=0):
    s = tensor.shape
    s_new = [*s[:dim],s[dim]*s[dim+1]]
    if dim+2<len(s):
        s_new+=s[dim+2:]
    return tensor.view(s_new)

# transposes two dimensions, flattens these in the mask and data tensor, and selects the masked indices
def transposed_mask_select(x,mask,tdims):
    x_prep    = mergeDim(   x.transpose(*tdims).contiguous(), dim=tdims[0])
    mask_prep = mergeDim(mask.transpose(*tdims).contiguous(), dim=tdims[0])
    x_new = torch.masked_select(x_prep,mask=mask_prep)
    return x_new


#def tensor_index_select(x,index):
#    IPython.embed()
#    index_flat = np.ravel_multi_index(index.numpy(),x.shape)
import numpy as np

def gpu_nms(polys, thres=0.3, K=100, precision=10000):
    from .nms_kernel import nms as nms_impl
    if len(polys) == 0:
        return np.array([], dtype='float32')
    p = polys.copy()
    #p[:,:8] *= precision
    ret = np.array(nms_impl(p, thres), dtype='int32')
    #ret[:,:8] /= precision
    return ret

def rbox_gpu_nms(polys, thres=0.3, K=100, precision=10000):
    from .nms_kernel import rbox_nms as nms_impl
    if len(polys) == 0:
        return np.array([], dtype='float32')
    p = polys.copy()
    #p[:,:8] *= precision
    ret = np.array(nms_impl(p, thres), dtype='int32')
    #ret[:,:8] /= precision
    return ret

def rbox_iou(quadboxes, query_quadboxes):
    from .nms_kernel import rbox_overlap as nms_impl
    quadboxes_shape = quadboxes.shape
    query_quadboxes_shape = query_quadboxes.shape

    iou_matrix = np.zeros([quadboxes_shape[0], quadboxes_shape[1], query_quadboxes_shape[1]])
    if quadboxes_shape[0] == 0 or query_quadboxes_shape[0] == 0 or query_quadboxes_shape[1] == 0 or quadboxes_shape[1] == 0 :
        print("erro!")
        return iou_matrix

    for i, (each_quad, each_query) in enumerate(zip(quadboxes, query_quadboxes)):
        flatt_iou, idx = nms_impl(each_quad, each_query)
        each_iou = iou_matrix[i]
        each_iou_flatten = np.reshape(each_iou, [-1])
        each_iou_flatten[idx] = flatt_iou
        iou_matrix[i] = np.reshape(each_iou_flatten, [quadboxes_shape[1], query_quadboxes_shape[1]])

        #iou_matrix.append(each_iou)#np.reshape(,[query_quadboxes[1], query_quadboxes_shape[1]]))
    print("rbox iou end")
    return iou_matrix.astype(np.float32)
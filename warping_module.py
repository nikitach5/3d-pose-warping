import tensorflow as tf
import numpy as np
import time
import tensorflow as tf
from parameters import params
import numpy as np



#for main body mask and head mask (rotation of plane)
#mask input shape=(3,:)
def rotation_estimation_3joint(joint1i,joint2i,joint3i,joint1f,joint2f,joint3f,mask):
  midi=(joint1i+joint2i)/2
  midf=(joint1f+joint2f)/2
  

  ai=joint1i-midi
  af=joint1f-midf
  bi=joint2i-midi
  bf=joint2f-midf
  ci=joint3i-midi
  cf=joint3f-midf
  
  
  scle=tf.norm(bi-ai)/tf.norm(bf-af)

  Mi=np.column_stack((ai*scle,bi*scle,ci*scle))
  Mf=np.column_stack((af,bf,cf))

  rotation_mat=tf.linalg.matmul((Mf),tf.linalg.pinv(Mi)) #s*R(ji-mi)=jf-mf
  midi=np.reshape(midi,(3,-1))

  mask=np.reshape(mask,(3,-1))
  
 
  midf=np.reshape(midf,(3,-1))
  maskf=np.matmul(rotation_mat,mask-midi)*scle+midf
  
  return maskf #mask's final coordinate

def rotation_estimation(joint1i,joint2i,joint1f,joint2f,mask):
  
  a=joint1i-joint2i
  b=joint1f-joint2f
  
  
  scle=np.linalg.norm([b[0],b[1]])/np.linalg.norm([a[0],a[1]])
  
  angle=tf.math.atan(b[1]/b[0])-tf.math.atan(a[1]/a[0])
  si=tf.math.sin(angle)
  co=tf.math.cos(angle)
  
  rotation_mat=np.array([[co,-si],[si,co]])
  
  mxy=np.array([mask[0,:],mask[1,:]])
  j2i=np.array([[joint2i[0]],[joint2i[1]]])
  j2f=np.array([[joint2f[0]],[joint2f[1]]])
  
  xymask=np.matmul(rotation_mat,mxy-j2i)*scle+j2f
  
  
  zmask=((xymask[0,:]-joint2f[0])*b[2]/b[0])+joint2f[2]
  
  zmask=np.reshape(zmask,(1,-1))
  maskf=np.concatenate((xymask,zmask),axis=0)
  
  return maskf

def warpingModule(mask,transform,joint):
    warped_mask=[]

    warped_mask.append(rotation_estimation(joint['lsho'],joint['lelb'],transform['lsho'],transform['lelb'],mask[0]))
    warped_mask.append(rotation_estimation(joint['rsho'],joint['relb'],transform['rsho'],transform['relb'],mask[1]))
    warped_mask.append(rotation_estimation(joint['lelb'],joint['lwri'],transform['lelb'],transform['lwri'],mask[2]))
    warped_mask.append(rotation_estimation(joint['relb'],joint['rwri'],transform['relb'],transform['rwri'],mask[3]))
    warped_mask.append(rotation_estimation(joint['lhip'],joint['lkne'],transform['lhip'],transform['lkne'],mask[4]))
    warped_mask.append(rotation_estimation(joint['rhip'],joint['rkne'],transform['rhip'],transform['rkne'],mask[5]))
    warped_mask.append(rotation_estimation(joint['lkne'],joint['lank'],transform['lkne'],transform['lank'],mask[6]))
    warped_mask.append(rotation_estimation(joint['rkne'],joint['rank'],transform['rkne'],transform['rank'],mask[7]))
    warped_mask.append(rotation_estimation(joint['lear'],joint['rear'],joint['reye'],transform['lear'],transform['rear'],transform['reye'],mask[8]))
    warped_mask.append(rotation_estimation(joint['neck'],joint['pelv'],joint['rsho'],transform['neck'],transform['pelv'],transform['rsho'],mask[9]))

    return wraped_mask


#Testing:
dt = time.time()

ji=np.array([3,-4,3])
jf=np.array([4.99,6.26,11.9])
j2i=np.array([-5,-7,-4])
j2f=np.array([3,-2,-6])
mask=np.array([[2.29,5.38,3.11,3],[0.07,2.43,3.92,2.5],[6.79,8.66,10,7.5]])
c=rotation_estimation(ji,jf,j2i,j2f,mask)
print(c)
df = time.time()

print('1 mask coordinate is generated in:',(df-dt)/4,'ms')


def build_coords(shape):
    xx, yy, zz = tf.meshgrid(tf.range(shape[1]), tf.range(shape[0]), tf.range(shape[2]))  # in image notation
    ww = tf.ones(xx.shape)
    coords = tf.concat([tf.expand_dims(tf.cast(a, tf.float32), -1) for a in [xx, yy, zz, ww]], axis=-1)
    return coords


# input in matrix notation
def transform_single(volume, transform, interpolation):
    volume = tf.transpose(volume, [1, 0, 2, 3])  # switch to image notation
    coords = build_coords(volume.shape[:3])
    coords_shape = coords.shape
    coords_reshaped = tf.reshape(coords, [-1, 4])
    pointers_reshaped = tf.linalg.matmul(transform, coords_reshaped, transpose_b=True)
    pointers = tf.reshape(tf.transpose(pointers_reshaped, [1, 0]), coords_shape)  # undo transpose_b
    pointers = pointers[:, :, :, :3]
    if interpolation == 'NEAREST':
        pointers = tf.cast(tf.math.round(pointers), dtype=tf.int32)
        with tf.device('/gpu:0'):
            res = tf.gather_nd(volume, pointers)
    elif interpolation == 'TRILINEAR':
        c3s = {}
        for c in [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]:
            c3s[c] = tf.gather_nd(volume, tf.cast(tf.floor(pointers), dtype=tf.int32) + c)
        d = pointers - tf.floor(pointers)
        c2s = {}
        for c in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            c2s[c] = c3s[(0,) + c] * (1 - d[:, :, :, 0:1]) + c3s[(1,) + c] * (d[:, :, :, 0:1])
        c1s = {}
        for c in [(0,), (1,)]:
            c1s[c] = c2s[(0,) + c] * (1 - d[:, :, :, 1:2]) + c2s[(1,) + c] * (d[:, :, :, 1:2])
        res = c1s[(0,)] * (1 - d[:, :, :, 2:3]) + c1s[(1,)] * (d[:, :, :, 2:3])
    else:
        raise ValueError
    return res


def volumetric_transform(volumes, transforms, interpolation='NEAREST'):
    return tf.map_fn(lambda x: transform_single(x[0], x[1], interpolation), (volumes, transforms), dtype=tf.float32,
                     parallel_iterations=128)


def warp_3d(vol_batch, masks_batch, transform_batch, reduce=True):
    n, h, w, d, c = vol_batch.get_shape().as_list()
    with tf.name_scope('warp_3d'):
        net = {}

        part_count = transform_batch.shape[1]

        net['bodypart_masks'] = masks_batch

        init_volume_size = (params['image_size'], params['image_size'], params['image_size'])
        z_scale = (d - 1) / (h - 1)
        v_scale = (h - 1) / init_volume_size[0]
        affine_mul = [[1, 1, 1 / z_scale, v_scale],
                      [1, 1, 1 / z_scale, v_scale],
                      [z_scale, z_scale, 1, v_scale * z_scale],
                      [1, 1, 1 / z_scale, 1]]
        affine_mul = np.array(affine_mul).reshape((1, 1, 4, 4))
        affine_transforms = transform_batch * affine_mul
        affine_transforms = tf.reshape(affine_transforms, (-1, 4, 4))

        expanded_tensor = tf.expand_dims(vol_batch, -1)
        multiples = [1, part_count, 1, 1, 1, 1]
        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        repeated_tensor = tf.reshape(tiled_tensor, (
            n * part_count, h, w, d, c))

        transposed_masks = tf.transpose(masks_batch, [0, 4, 1, 2, 3])
        reshaped_masks = tf.reshape(transposed_masks, [n * part_count, h, w, d])
        repeated_tensor = repeated_tensor * tf.expand_dims(reshaped_masks, axis=-1)

        net['masked_bodyparts'] = repeated_tensor
        warped = volumetric_transform(repeated_tensor, affine_transforms, interpolation='TRILINEAR')
        net['masked_bodyparts_warped'] = warped

        res = tf.reshape(warped, [-1, part_count, h, w, d, c])
        res = tf.transpose(res, [0, 2, 3, 4, 1, 5])
        if reduce:
            res = tf.reduce_max(res, reduction_indices=[-2])
        return res, net


def warp_2d_3d(vol_batch, masks_batch, transform_batch, reduce=True):
    n, h, w, d, c = vol_batch.get_shape().as_list()
    with tf.name_scope('warp_2d_3d'):
        net = {}
        part_count = transform_batch.shape[1]

        # MASKS 3D

        net['bodypart_masks'] = masks_batch

        img_batch = tf.reshape(vol_batch, (n, h, w, d * c))

        init_image_size = (params['image_size'], params['image_size'])
        affine_mul = [1, 1, init_image_size[0] / h,
                      1, 1, init_image_size[1] / w,
                      1, 1]
        affine_mul = np.array(affine_mul).reshape((1, 1, 8))

        expanded_tensor = tf.expand_dims(img_batch, -1)
        multiples = [1, part_count, 1, 1, 1]
        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        repeated_tensor = tf.reshape(tiled_tensor, tf.shape(img_batch) * np.array([part_count, 1, 1, 1]))

        affine_transforms = transform_batch / affine_mul
        affine_transforms = tf.reshape(affine_transforms, (-1, 8))

        transposed_masks = tf.transpose(masks_batch, [0, 3, 1, 2])
        reshaped_masks = tf.reshape(transposed_masks, [n * part_count, h, w])
        repeated_tensor = repeated_tensor * tf.expand_dims(reshaped_masks, axis=-1)
        warped = tf.contrib.image.transform(repeated_tensor, affine_transforms)
        res = tf.reshape(warped, [-1, part_count, h, w, d * c])
        res = tf.transpose(res, [0, 2, 3, 1, 4])
        if reduce:
            res = tf.reduce_max(res, reduction_indices=[-2])

        res = tf.reshape(res, (n, h, w, d, -1))
        return res, net


def residual_unit_3d(x):
    filters = x.shape[-1]
    r = x
    for i in range(2):
        r = group_norm(r)
        r = tf.nn.relu(r)
        r = tf.layers.conv3d(r, filters, kernel_size=3, padding='SAME')
    return x + r


def tf_pose_map_3d(poses, shape):
    y = tf.unstack(poses, axis=1)
    y[0], y[1] = y[1], y[0]
    poses = tf.stack(y, axis=1)
    image_size = tf.constant(params['image_size'], tf.float32)
    shape = tf.constant(shape, tf.float32)
    sigma = tf.constant(6, tf.float32)
    poses = tf.unstack(poses, axis=0)
    pose_mapss = []
    for pose in poses:
        pose = pose / image_size * shape[:, tf.newaxis]
        joints = tf.unstack(pose, axis=-1)
        pose_maps = []
        for joint in joints:
            xx, yy, zz = tf.meshgrid(tf.range(shape[0]), tf.range(shape[1]), tf.range(shape[2]), indexing='ij')
            mesh = tf.cast(tf.stack([xx, yy, zz]), dtype=tf.float32)
            pose_map = mesh - joint[:, tf.newaxis, tf.newaxis, tf.newaxis]
            pose_map = pose_map / shape[:, tf.newaxis, tf.newaxis, tf.newaxis] * image_size
            pose_map = tf.exp(-tf.reduce_sum(pose_map ** 2, axis=0) / (2 * sigma ** 2))
            pose_maps.append(pose_map)
        pose_map = tf.stack(pose_maps, axis=-1)
        if params['2d_3d_pose']:
            pose_map = tf.reduce_max(pose_map, axis=2, keepdims=True)
            pose_map = tf.tile(pose_map, [1, 1, params['depth'], 1])
        pose_mapss.append(pose_map)
    return tf.stack(pose_mapss, axis=0)


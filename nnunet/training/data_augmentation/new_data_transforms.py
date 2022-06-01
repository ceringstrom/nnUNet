import numpy as np
import scipy.ndimage.interpolation

from scipy.stats import halfnorm
from skimage.measure import label
from skimage.morphology import convex_hull_image
from scipy.stats import nakagami
import scipy.misc
from batchgenerators.transforms import AbstractTransform
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from skimage.draw import polygon


def find_mask(img, threshold=1/800, index=-2):
    v = img
    # print(img.shape)
    diff_mean = (np.clip(v, 0, 1))
    # find connected components after threshold
    cc = label(diff_mean > threshold)
    v, c = np.unique(cc, return_counts=True)
    # find second largest connect component (largest is the background)
    second_largest_component = v[c.tolist().index(sorted(c)[index])]

    # take convex hull to remove small gaps
    # noinspection PyTypeChecker
    return convex_hull_image(np.where(cc == second_largest_component, 1, 0))

def us_zoom_augmentation(sample_data, sample_seg, zoom=0.5, b=0):
    zoom_scale = halfnorm.rvs(loc=1, scale=zoom)

    # print("ZOOMIN")
    # print(sample_data.shape)
    # print(sample_seg.shape)
    # print(sample_data[0,0], sample_seg[0,0])
    # print("uniqueee")
    # print(np.unique(sample_seg))
    # print(sample_seg.dtype)
    np.unique(sample_seg)

    sample_data = np.squeeze(sample_data)
    sample_seg = np.squeeze(sample_seg)

    # matplotlib.image.imsave("testog_{}.png".format(b), np.squeeze(sample_data), cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
    # matplotlib.image.imsave("testogseg_{}.png".format(b), np.squeeze(sample_seg), cmap=plt.get_cmap('gray'), vmin=-1, vmax=8)

    scaled_img = sample_data / np.max(sample_data)
    scale = zoom_scale
    mask = find_mask(scaled_img)
    total = np.product(mask.shape)
    covering = mask.sum() / total
    com = scipy.ndimage.measurements.center_of_mass(mask)
    # print("com")

    if covering < .15:
        mask = find_mask(scaled_img, 1/800 / 2)
    if covering > .9 or com[0] < mask.shape[0]/4:
        mask = find_mask(scaled_img, 1/800, -1)

    zoomed_img = scipy.ndimage.zoom(sample_data, scale)
    zoomed_seg = np.round(scipy.ndimage.zoom(sample_seg, scale))

    img_top = np.nonzero(sample_data[:, int(com[1])])[0][0]

    com = scipy.ndimage.measurements.center_of_mass(mask)
    r_offset = int((scale-1) *  img_top)
    c_offset = min(max(0, int(scale * com[1] - sample_data.shape[1] / 2)), zoomed_img.shape[1] - sample_data.shape[1] )

    new_image = mask * zoomed_img[r_offset:r_offset + sample_data.shape[0], c_offset:c_offset + sample_data.shape[1]]
    new_seg = mask * zoomed_seg[r_offset:r_offset + sample_data.shape[0], c_offset:c_offset + sample_data.shape[1]]
    new_seg[np.where(sample_seg==-1)] = -1

    new_image = np.expand_dims(new_image, axis=0)
    new_seg = np.expand_dims(new_seg, axis=0)

    # matplotlib.image.imsave("testnew_{}.png".format(b), np.squeeze(sample_data), cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
    # matplotlib.image.imsave("testnewseg_{}.png".format(b), np.squeeze(sample_seg), cmap=plt.get_cmap('gray'), vmin=-1, vmax=8)

    return new_image, new_seg

def us_depth_augmentation(sample_data, sample_seg, zoom=0.5, b=0):
    zoom_scale = halfnorm.rvs(loc=1, scale=zoom)

    sample_data = np.squeeze(sample_data)
    sample_seg = np.squeeze(sample_seg)

    # matplotlib.image.imsave("testog_{}.png".format(b), np.squeeze(sample_data), cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
    # matplotlib.image.imsave("testogseg_{}.png".format(b), np.squeeze(sample_seg), cmap=plt.get_cmap('gray'), vmin=-1, vmax=8)

    scaled_img = sample_data / np.max(sample_data)
    scale = zoom_scale
    mask = find_mask(scaled_img)
    total = np.product(mask.shape)
    covering = mask.sum() / total
    com = scipy.ndimage.measurements.center_of_mass(mask)
    # print("com")

    if covering < .15:
        mask = find_mask(scaled_img, 1/800 / 2)
    if covering > .9 or com[0] < mask.shape[0]/4:
        mask = find_mask(scaled_img, 1/800, -1)

    zoomed_img = scipy.ndimage.zoom(sample_data, scale)
    zoomed_seg = np.round(scipy.ndimage.zoom(sample_seg, scale))

    com = scipy.ndimage.measurements.center_of_mass(mask)
    r_offset = min(max(0, int(scale * com[0] - sample_data.shape[0] / 2)), zoomed_img.shape[0] - sample_data.shape[0] )
    c_offset = min(max(0, int(scale * com[1] - sample_data.shape[1] / 2)), zoomed_img.shape[1] - sample_data.shape[1] )

    new_image = mask * zoomed_img[r_offset:r_offset + sample_data.shape[0], c_offset:c_offset + sample_data.shape[1]]
    new_seg = mask * zoomed_seg[r_offset:r_offset + sample_data.shape[0], c_offset:c_offset + sample_data.shape[1]]
    new_seg[np.where(sample_seg==-1)] = -1

    new_image = np.expand_dims(new_image, axis=0)
    new_seg = np.expand_dims(new_seg, axis=0)

    return new_image, new_seg

def get_scanlines(img):
    left_line = []
    right_line = []

    start_row = 0
    h, w = img.shape
    min_col = w
    for i in range(h):
        row = img[i, :]
        if(np.count_nonzero(row)) > 1:
            nonzero = row.nonzero()
            if start_row==0:
                start_row = i
            if nonzero[0][0] < min_col:
                min_col = nonzero[0][0]
            if nonzero[0][0] > min_col + 5:
                break
            left_line.append(nonzero[0][0])
            right_line.append(nonzero[0][-1])
    y = np.arange(start_row, start_row+len(left_line))
    left = np.polyfit(np.array(left_line),y,1)
    right = np.polyfit(np.array(right_line),y,1)
    xi = (left[1]-right[1])/(right[0]-left[0])
    yi = left[0] * xi + left[1]
    slopes = (left, right)
    if yi > 0:
        yi = 0
        # matplotlib.image.imsave("testshad.png", np.squeeze(img), cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
    #print(angle)                                                       

    radius = img.shape[0] - yi
    center = left_line[0] + (right_line[0] - left_line[0])/2

    return center, radius, -yi, slopes

def get_triangle_mask(img_shape, center, radius, buffer, slopes, width=[0, 0.2]):
    h, w = img_shape
    
    m1 = slopes[0][0]
    m2 = slopes[1][0]
    theta_low = min(np.arctan(1/m1), np.arctan(1/m2))
    theta_high = max(np.arctan(1/m1), np.arctan(1/m2))
    
    if width[1] > theta_high - theta_low:
        width[1] = theta_high - theta_low
    width = np.random.uniform(low=width[0], high=width[1], size=1)
    position = np.random.uniform(low=theta_low+width/2, high=theta_high-width/2, size=1)
    low_slope = 1/np.tan(position-width/2)
    high_slope = 1/np.tan(position+width/2)
    low_intercept = -buffer - low_slope * center
    high_intercept = -buffer - high_slope * center
    lines = [(low_slope[0], low_intercept[0]), (high_slope[0], high_intercept[0])]
    contour_rr = []
    contour_cc = []
    for (slope, intercept) in lines:
        bottom_intercept = (h-intercept) / slope
        if intercept >= 0 and intercept <= h:
            contour_cc.append(0)
            contour_rr.append(intercept)
        elif(bottom_intercept >= 0 and bottom_intercept <= w):
            contour_cc.append(bottom_intercept)
            contour_rr.append(h)
        else:
            contour_cc.append(w)
            contour_rr.append(slope * w + intercept)
    contour_cc.append(-lines[1][1] / lines[1][0])
    contour_rr.append(0)

    contour_cc.append(-lines[0][1] / lines[0][0])
    contour_rr.append(0)

    mask = np.zeros((h, w))
    rr, cc = polygon(contour_rr, contour_cc)
    mask[rr, cc] = 1
    return mask

def get_bounded_nakagami(length, shape, scale, min_val=0, max_val=255):
    vals = np.zeros(length)
    for i in range(length):
        val = -1
        while val < min_val or val > max_val:
            val = nakagami.rvs(shape, scale=scale, size=1)
        vals[i] = val
    return vals

def shadow_augmentation(sample_data, sample_seg):
    img = np.squeeze(sample_data)
    sample_seg = np.squeeze(sample_seg)
    # try:
        # cv2.imwrite("test_{}.png".format(i), (img*255/ np.max(img)).astype("int8"))
    center, radius, buffer, slopes = get_scanlines(img*255)
    top = (img[:, int(center)]!=0).argmax() + int(buffer)
    mask = get_triangle_mask(img.shape, center, radius, int(buffer), slopes)
    num_pix = int(mask.sum())
    nak = get_bounded_nakagami(num_pix, 0.202, 189.3, min_val=0, max_val=255) / 255 * np.max(img)
    masked_image = img.copy()
    sample_seg[mask==1] = 0
    masked_image[np.where(mask != 0)] = nak
    outer_mask = find_mask(img, index=-2)
    masked_image = masked_image * outer_mask
    sample_seg[outer_mask==0] = -1
    # except Exception as e:
        # masked_image = img
    # cv2.imwrite("test2_{}.png".format(i), (masked_image*255/ np.max(img)).astype("int8"))
    masked_image = np.expand_dims(masked_image, axis=0)
    sample_seg = np.expand_dims(sample_seg, axis=0)
    return masked_image, sample_seg
    

def gain_augmentation(sample_data, gain_range=(0.5, 2), regions=10):
    img = np.squeeze(sample_data)
    center, radius, buffer, slopes = get_scanlines(img*255)
    maxval = np.max(img)
    gains = np.random.uniform(gain_range[0], gain_range[1], regions)
    # print(gains)
    nz_indices = np.nonzero(img)
    # print(np.sum(img==0.0))
    # cv2.imwrite("tgain.png", (img==0).astype("uint8")*255)
    radii = np.sqrt(np.square(nz_indices[0] + buffer) + np.square(nz_indices[1] - center))
    lower_limit = np.min(radii)
    upper_limit = np.max(radii)
    region_size = (upper_limit-lower_limit) / regions
    gain_array = np.zeros(radii.shape)
    # print((gain_array.dtype))
    current_pos = lower_limit
    for i in range(regions):
        gain_array[np.where(radii > current_pos)] = gains[i]
        current_pos += region_size
    # print("gains", np.max(gain_array), np.min(gain_array))
    # print(img.dtype)
    # img[nz_indices] = gain_array
    # cv2.imwrite("gain.png", (img*255/np.max(gains)).astype("uint8"))
    # print("monmax")
    # print(np.unique((img*255/np.max(gains)).astype("uint8"))) #, np.min((img*255/np.max(gains)).astype("uint8")))
    img[nz_indices] = gain_array * img[nz_indices]
    # print(np.unique(img).shape)
    img[img > maxval] = maxval

    img = np.expand_dims(img, axis=0)

    return img


class UsZoomTransform(AbstractTransform):
    def __init__(self, data_key="data", label_key="seg", p_per_sample=1, zoom=0.5):
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample
        self.zoom = zoom

    def __call__(self, **data_dict):
        data  = data_dict.get(self.data_key).copy()
        seg = data_dict.get(self.label_key).copy()
        
        for b in range(len(data)):
            if np.random.uniform() < self.p_per_sample:
                # matplotlib.image.imsave("testog_{}.png".format(b), np.squeeze(data[b]), cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
                # matplotlib.image.imsave("testogseg_{}.png".format(b), np.squeeze(seg[b]), cmap=plt.get_cmap('gray'), vmin=-1, vmax=8)
                sample_seg = None
                if seg is not None:
                    sample_seg = seg[b]
                maxval = np.max(sample_seg)
                # print(np.max(data[b]))
                # print(np.min(data[b]))
                # print(data[b].shape)
                # print(np.min((data[b]-np.min(data[b]))))
                # print(np.max(data[b]), np.min(data[b]))
                # print(b)
                # print("nonzero")
                # print(np.count_nonzero(data[b]))
                # print(np.count_nonzero(data[b].astype("int8")))
                # print(data[b].shape)
                # print("SAVING")
            
                ret_val = us_zoom_augmentation(data[b], sample_seg, zoom=self.zoom, b=b)
                data[b] = ret_val[0]
                
                if seg is not None:
                    seg_temp = ret_val[1]
                    seg_temp[data_dict.get(self.label_key)[b]==-1] = -1
                    seg_temp[seg_temp<-1] = -1
                    seg_temp[seg_temp > maxval] = maxval
                    seg[b] = seg_temp

                mask = seg[b] >= 0
                # data[b][mask] = (data[b][mask] - data[b][mask].mean()) / (data[b][mask].std() + 1e-8)
                data[b][mask == 0] = 0
                # print(np.max(data[b]), np.min(data[b]))
    
                # matplotlib.image.imsave("zmask_{}.png".format(b), np.squeeze(mask), cmap=plt.get_cmap('gray'), vmin=0, vmax=3)
                # matplotlib.image.imsave("testnew_{}.png".format(b), np.squeeze(data[b]), cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
                # matplotlib.image.imsave("testnewseg_{}.png".format(b), np.squeeze(seg[b]), cmap=plt.get_cmap('gray'), vmin=-1, vmax=8)

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg

        return data_dict


class ShadowTransform(AbstractTransform):
    def __init__(self, data_key="data", label_key="seg", p_per_sample=1):
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        data  = data_dict.get(self.data_key).copy()
        seg = data_dict.get(self.label_key).copy()

        for b in range(len(data)):
            if np.random.uniform() < self.p_per_sample:
                # print(np.sum(data[b] >= 0))
                # matplotlib.image.imsave("testshadbef{}.png".format(b), np.squeeze(data[b]), cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
                # matplotlib.image.imsave("testshadbefseg{}.png".format(b), np.squeeze(seg[b]), cmap=plt.get_cmap('gray'), vmin=-1, vmax=6)
                try:
                    ret_val, ret_seg = shadow_augmentation(data[b], seg[b])

                    seg[b] = ret_seg
                    mask = np.squeeze(seg[b] >= 0)
                    # print("before thing")
                    # print(np.sum(ret_val > 0))
                    ret_val[0][mask==0] = 0
                    # matplotlib.image.imsave("testshadaf{}.png".format(b), np.squeeze(ret_val), cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
                    # matplotlib.image.imsave("testshadafseg{}.png".format(b), np.squeeze(seg[b]), cmap=plt.get_cmap('gray'), vmin=-1, vmax=6)
                    # print("mask stuff")
                    # print(np.unique(seg))
                    # print(np.sum(mask), np.sum(old_mask), np.sum(ret_val > 0), np.sum(data[b]>0))
                    # print(ret_val[0][mask].mean())
                    # print(ret_val[0][mask].std() + 1e-8)
                    # print(ret_val[0].shape, mask.shape)
                    # ret_val[0][mask] = (ret_val[0][mask] - ret_val[0][mask].mean()) / (ret_val[0][mask].std() + 1e-8)
                    
                    # print(mask.shape)
                    # print(ret_val.shape)
                    # print(np.max(ret_val), np.min(ret_val))
                except Exception as e:
                    print("problem")
                    print(e)
                else:
                    data[b] = ret_val

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg

        return data_dict

class DepthTransform(AbstractTransform):
    def __init__(self, data_key="data", label_key="seg", p_per_sample=1):
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample
        self.zoom = 0.5

    def __call__(self, **data_dict):
        data  = data_dict.get(self.data_key).copy()
        seg = data_dict.get(self.label_key).copy()
        
        for b in range(len(data)):
            if np.random.uniform() < self.p_per_sample:
                # matplotlib.image.imsave("testog_{}.png".format(b), np.squeeze(data[b]), cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
                # matplotlib.image.imsave("testogseg_{}.png".format(b), np.squeeze(seg[b]), cmap=plt.get_cmap('gray'), vmin=-1, vmax=8)
                sample_seg = None
                if seg is not None:
                    sample_seg = seg[b]
                maxval = np.max(sample_seg)
                # print(np.max(data[b]))
                # print(np.min(data[b]))
                # print(data[b].shape)
                # print(np.min((data[b]-np.min(data[b]))))
                # print(np.max(data[b]), np.min(data[b]))
                # print(b)
                # print("nonzero")
                # print(np.count_nonzero(data[b]))
                # print(np.count_nonzero(data[b].astype("int8")))
                # print(data[b].shape)
                # print("SAVING")
            
                # matplotlib.image.imsave("testimog_{}.png".format(b), np.squeeze(data[b]), cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
                # matplotlib.image.imsave("testsegog_{}.png".format(b), np.squeeze(seg[b]), cmap=plt.get_cmap('gray'), vmin=-1, vmax=8)

                ret_val = us_zoom_augmentation(data[b], sample_seg, zoom=self.zoom, b=b)
                data[b] = ret_val[0]
                
                if seg is not None:
                    seg_temp = ret_val[1]
                    seg_temp[data_dict.get(self.label_key)[b]==-1] = -1
                    seg_temp[seg_temp<-1] = -1
                    seg_temp[seg_temp > maxval] = maxval
                    seg[b] = seg_temp

                mask = seg[b] >= 0
                # data[b][mask] = (data[b][mask] - data[b][mask].mean()) / (data[b][mask].std() + 1e-8)
                data[b][mask == 0] = 0

                # matplotlib.image.imsave("testimnew_{}.png".format(b), np.squeeze(data[b]), cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
                # matplotlib.image.imsave("testsegnew{}.png".format(b), np.squeeze(seg[b]), cmap=plt.get_cmap('gray'), vmin=-1, vmax=8)
                # print(np.max(data[b]), np.min(data[b]))
    
                # matplotlib.image.imsave("zmask_{}.png".format(b), np.squeeze(mask), cmap=plt.get_cmap('gray'), vmin=0, vmax=3)
                # matplotlib.image.imsave("testnew_{}.png".format(b), np.squeeze(data[b]), cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
                # matplotlib.image.imsave("testnewseg_{}.png".format(b), np.squeeze(seg[b]), cmap=plt.get_cmap('gray'), vmin=-1, vmax=8)

        data_dict[self.data_key] = data
        if seg is not None:
            data_dict[self.label_key] = seg

        return data_dict

class GainTransform(AbstractTransform):
    def __init__(self, data_key="data", label_key="seg", p_per_sample=1):
        self.data_key = data_key
        self.label_key = label_key
        self.p_per_sample = p_per_sample

    def __call__(self, **data_dict):
        data  = data_dict.get(self.data_key).copy()
        seg = data_dict.get(self.label_key).copy()

        for b in range(len(data)):
            if np.random.uniform() < self.p_per_sample:
                # print(np.sum(data[b] >= 0))
                # matplotlib.image.imsave("testshadbef{}.png".format(b), np.squeeze(data[b]), cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
                # matplotlib.image.imsave("testshadbefseg{}.png".format(b), np.squeeze(seg[b]), cmap=plt.get_cmap('gray'), vmin=-1, vmax=6)
                try:
                    # matplotlib.image.imsave("testimog_{}.png".format(b), np.squeeze(data[b]), cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
                    # matplotlib.image.imsave("testsegog_{}.png".format(b), np.squeeze(seg[b]), cmap=plt.get_cmap('gray'), vmin=-1, vmax=8)

                    ret_val = gain_augmentation(data[b])
                    # matplotlib.image.imsave("testshadaf{}.png".format(b), np.squeeze(ret_val), cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
                    # matplotlib.image.imsave("testshadafseg{}.png".format(b), np.squeeze(seg[b]), cmap=plt.get_cmap('gray'), vmin=-1, vmax=6)
                    # print("mask stuff")
                    # print(np.unique(seg))
                    # print(np.sum(mask), np.sum(old_mask), np.sum(ret_val > 0), np.sum(data[b]>0))
                    # print(ret_val[0][mask].mean())
                    # print(ret_val[0][mask].std() + 1e-8)
                    # print(ret_val[0].shape, mask.shape)
                    # ret_val[0][mask] = (ret_val[0][mask] - ret_val[0][mask].mean()) / (ret_val[0][mask].std() + 1e-8)
                    
                    # print(mask.shape)
                    # print(ret_val.shape)
                    # print(np.max(ret_val), np.min(ret_val))
                    
                except Exception as e:
                    print("problem")
                    print(e)
                else:
                    data[b] = ret_val
                # matplotlib.image.imsave("testimnew_{}.png".format(b), np.squeeze(data[b]), cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
                # matplotlib.image.imsave("testsegnew_{}.png".format(b), np.squeeze(seg[b]), cmap=plt.get_cmap('gray'), vmin=-1, vmax=8)

        data_dict[self.data_key] = data

        return data_dict


class NormalizeTransform(AbstractTransform):
    def __init__(self, data_key="data", label_key="seg"):
        self.data_key = data_key
        self.label_key = label_key

    def __call__(self, **data_dict):
        data  = data_dict.get(self.data_key).copy()
        seg = data_dict.get(self.label_key).copy()

        for b in range(len(data)):
            mask = seg[b] >= 0
            data[b][mask] = data[b][mask] / 255
            # (data[b][mask] - data[b][mask].mean()) / (data[b][mask].std() + 1e-8)
        
        data_dict[self.data_key] = data

        return data_dict
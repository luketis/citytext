import numpy as np
from scipy.special import softmax


def filter_cut_text_mask(df, bbox_col, img_shape, tol=5):

    def calc_bbox_stats(bbox):
        if isinstance(bbox, float):
            return np.nan, np.nan, np.nan, np.nan

        xs, ys = zip(*eval(bbox))

        return min(xs), max(xs), min(ys), max(ys)

    det_bbox_stats = df[bbox_col].transform(calc_bbox_stats)
    minx, maxx = det_bbox_stats.str[0], det_bbox_stats.str[1] 
    miny, maxy = det_bbox_stats.str[2], det_bbox_stats.str[3]

    return (minx > tol) & (maxx < img_shape[1] - tol)


def filter_mask(df, rec_conf=0.8, min_len=3,
                filter_stop=True,
                filter_numeric=True,
                filter_cut_text=True, cut_text_tol=5, 
                txt_col='rec_txt', bbox_col='real_detected_bbox',
                stop_words=('the', 'and', 'was', 'are', 'not', 'more', 'any')):

    stop_mask = df[txt_col].isin(stop_words)
    conf_mask = df.rec_conf > rec_conf
    min_len_3 = lambda txt_col: df[txt_col].str.len() > min_len

    numeric_mask = lambda txt_col: df[txt_col].str.isnumeric().fillna(True)

    mask = conf_mask & min_len_3(txt_col)

    if filter_stop:
        mask = mask & ~stop_mask

    if filter_numeric:
        mask = mask & ~numeric_mask(txt_col)

    if filter_cut_text:
        cut_text_mask = filter_cut_text_mask(df, bbox_col, (640, 640), tol=cut_text_tol)
        mask = mask & cut_text_mask

    return mask


def classify_topics(df, embd_dict, encoder, topic_descriptions, 
                    consider_ties=True, min_thresh=0.6):
    all_embds = []

    invalid_idx = []

    for i, txt in enumerate(df.rec_txt):
        if txt not in embd_dict or txt is None or isinstance(txt, float):
            all_embds += [np.array([0.0]*(len(embd_dict[list(embd_dict.keys())[0]]) - 1) + [1.0], dtype=np.float32)]
            invalid_idx += [i]
        else:
            all_embds += [embd_dict[txt]]

    all_embds = np.array(all_embds)

    topic_labels = list(topic_descriptions.keys())
    topic_embds = encoder.encode([topic_descriptions[topic] for topic in topic_labels]).cpu().numpy()

    topic_similarities = np.array(encoder.cos_dist_cartesian(all_embds, topic_embds))

    topic_cls_idx = np.argmax(topic_similarities, axis=1)

    for idx in invalid_idx:
        topic_cls_idx[idx] = -1

    if not consider_ties:
        topic_similarities = softmax(topic_similarities, axis=1)

        topic_cls_idx[topic_similarities.max(axis=1) < min_thresh] = -1

    df['topic_cls_label'] = [topic_labels[idx] for idx in topic_cls_idx]

    return df

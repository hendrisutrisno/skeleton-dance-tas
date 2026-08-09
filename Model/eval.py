import numpy as np
from pathlib import Path
import pandas as pd
from loguru import logger
from action_dict import action_label
from visual import color_list, before_after_chart, chart, comparsion_chart

fps = 30
def argrelmax(prob: np.ndarray, threshold: float = 0.7):
    """
    Calculate arguments of relative maxima.
    prob: np.array. boundary probability maps distributerd in [0, 1]
    prob shape is (T)
    ignore the peak whose value is under threshold

    Return:
        Index of peaks for each batch
    """
    # ignore the values under threshold
    prob[prob < threshold] = 0.0

    # calculate the relative maxima of boundary maps
    # treat the first frame as boundary
    peak = np.concatenate(
        [
            np.ones((1), dtype=np.bool),
            (prob[:-2] < prob[1:-1]) & (prob[2:] < prob[1:-1]),
            np.zeros((1), dtype=np.bool),
        ],
        axis=0,
    )

    peak_idx = np.where(peak)[0].tolist()

    return peak_idx


def read_file(path):
    label_path = Path(path)
    if label_path.suffix == ".csv":
        try:
            label = pd.read_csv(label_path)
        except FileNotFoundError as e:
            logger.error(f"label file not found:{e}")
            return
        return label
    else:
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError as e:
            logger.error(f"label file not found:{e}")
            return
        return lines

def framewise_label(start, end, acts):
    f_label = []
    for i, (s, e) in enumerate(zip(start, end)):
        for _ in range(int(s * (fps)), int(e * (fps))):
            f_label.append(action_label[acts[i]])
    return f_label  

def boundary_label(start, end, acts):
    b_label = []
    for i, (s, e) in enumerate(zip(start, end)):
        for _ in range(int(s * (fps)), int(e * (fps))):
            b_label.append(0)
        b_label[-1] = 1
    #b_label[0] = 1
    b_label[-1] = 0
    return b_label

def time_label(preds):
    labels = [preds[0]]
    starts = [0]
    ends = []
    last_label = preds[0]
    for i in range(len(preds)):
        if preds[i] != last_label:
            ends.append(i)
            labels.append(preds[i])
            starts.append(i)
        last_label = preds[i]
    ends.append(i)
    return labels, starts, ends

def levenstein(p, y, norm=True):
    m_row = len(p)
    n_col = len(y)
    d = np.zeros([m_row+1, n_col+1], np.float32)
    for i in range(m_row+1):
        d[i, 0] =i
    for i in range(n_col+1):
        d[0, i] =i
    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                d[i, j] = d[i-1, j-1]
            else:
                d[i, j] = min(d[i-1, j]+1, d[i, j-1]+1, d[i-1, j-1]+1)
    if norm:
        score = (1 - d[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = d[-1, -1]
    return score

def edit_score(pred, label, bg_action, norm=True):
    p, _, _ = time_label(pred)
    label = [s for s in label if s not in bg_action]
    p = [s for s in p if s not in bg_action]
    score = levenstein(p, label, norm)
    return score

def f_score(pred, label, s_label, e_label, overlap, bg_action):
    p, p_start, p_end = time_label(pred)

    idx = np.array([i for i, a in enumerate(p) if a not in bg_action], dtype=int)
    p = [p[i] for i in idx]
    p_start = [p_start[i] for i in idx]
    p_end   = [p_end[i] for i in idx]

    idx = np.array([i for i, a in enumerate(label) if a not in bg_action], dtype=int)
    label = [label[i] for i in idx]
    s_label = [s_label[i] for i in idx]
    e_label  = [e_label[i] for i in idx]

    if len(label) == 0:
        return 0.0, float(len(p)), 0.0
    if len(p) == 0:
        return 0.0, 0.0, float(len(label))

    tp = 0
    fp = 0
    hits = np.zeros(len(label))
    for j in range(len(p)):
        intersection = np.minimum(p_end[j] / fps, e_label) - np.maximum(p_start[j] / fps, s_label)
        union = np.maximum(p_end[j] / fps, e_label) - np.minimum(p_start[j] / fps, s_label)
        iou = (1. * intersection / union) * ([p[j] == label[x] for x in range(len(label))])
        idx = np.array(iou).argmax()
        if iou[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(label) - sum(hits)
    return float(tp), float(fp), float(fn)

def eval(pred_path, label_path, quality_path, bg_action = []):
    colors = color_list()
    pred_folder = Path(pred_path)
    #refine_dir = Path(pred_path) / "refinement"
    try:
        pred_files = [file for file in pred_folder.iterdir() if file.suffix == '.txt']
        #r_files = [file for file in refine_dir.iterdir() if file.suffix == '.txt']
    except FileNotFoundError as e:
        logger.error(f"folder not found: {e}")
        return

    bg_number = [action_label[i] for i in bg_action]

    label_data = read_file(label_path)
    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
    edit = 0
    edit_s = 0
    all_acc = 0
    pred_files.sort()
    #r_files.sort()

    #for pred, compare in zip(r_files, pred_files):
    for pred in pred_files:
        correct = 0
        total = 0
        video_id = int(pred.stem.split('/')[0])
        pred_lines = read_file(pred)
        if not pred_lines:
            print('no pred_line')
            continue
        pred_data = pred_lines[0].split()
        pred_num = [action_label[i] for i in pred_data]

        #c_lines = read_file(compare)
        #if not c_lines:
            #continue
        #c_data = c_lines[0].split()
        #c_num = [action_label[i] for i in c_data]

        actions = label_data[label_data["video id"] == video_id]
        start = np.array(actions["start"])
        end = np.array(actions["end"])
        acts = np.array(actions["action label"])

        y_label = framewise_label(start, end, acts)
        for i in range(min(len(y_label), len(pred_num))):
            if y_label[i] not in bg_number:
                total += 1
                if y_label[i] == pred_num[i]:
                    correct += 1

        edit = edit_score(pred_data, acts, bg_action)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(pred_data, acts, start, end, overlap[s], bg_action)
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
        
        accuraccy = 100 * (float(correct) / total) if total > 0 else 0
        all_acc += accuraccy
        chart(np.array(y_label), np.array(pred_num), colors, quality_path/ f"{video_id}.png")
        #before_after_chart(np.array(y_label), np.array(c_num), np.array(pred_num), colors, quality_path / f"{video_id}.png", refine)

        logger.info("Acc_%s: %.4f" %(video_id, accuraccy))
        logger.info("Edit_%s: %.4f" %(video_id, edit))
        edit_s += edit
    
    f1s = []
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])
        f1 = 2. * (precision * recall) / (precision + recall) if (precision+recall) > 0 else 0
        f1 = f1 * 100
        logger.info('F1@%0.2f: %.4f' % (overlap[s], f1))
        f1s.append(f1)
    
    all_acc = all_acc / len(pred_files) if all_acc > 0 else 0
    edit_s = edit_s/ len(pred_files) if edit_s > 0 else 0
    return all_acc, edit_s, f1s
    
def avg_eval(train_path, train_label, test_path, test_label):
    pred_folder = Path(train_path)
    try:
        train_files = [file for file in pred_folder.iterdir() if file.suffix == '.txt']
    except FileNotFoundError as e:
        logger.error(f"folder not found: {e}")
        return
    train_files.sort()

    pred_folder = Path(test_path)
    try:
        test_files = [file for file in pred_folder.iterdir() if file.suffix == '.txt']
    except FileNotFoundError as e:
        logger.error(f"folder not found: {e}")
        return
    test_files.sort()

    train = read_file(train_label)
    test = read_file(test_label)
    overlap = [.1, .25, .5]
    classes = list(action_label.values())
    action_acc = [0] * len(action_label)
    action_total = [0] * len(action_label)
    action_correct = [0] * len(action_label)
    action_tp1 = np.zeros((len(action_label),3))
    action_fp1 = np.zeros((len(action_label),3))
    action_fn1 = np.zeros((len(action_label),3))

    for pred in train_files:
        video_id = int(pred.stem.split('/')[0])
        pred_lines = read_file(pred)
        if not pred_lines:
            print('no pred_line')
            continue
        pred_data = pred_lines[0].split()
        pred_num = [action_label[i] for i in pred_data]

        actions = train[train["video id"] == video_id]
        start = np.array(actions["start"])
        end = np.array(actions["end"])
        acts = np.array(actions["action label"])
        y_label = framewise_label(start, end, acts)

        for ind, label in enumerate(classes):
            total = 0
            correct = 0
            for i in range(min(len(y_label), len(pred_num))):
                if y_label[i] == label:
                    total += 1
                    if y_label[i] == pred_num[i]:
                        correct += 1
            action_total[ind] += total
            action_correct[ind] += correct

            pre_SIL = [x for i, x in enumerate(list(action_label.keys())) if i != ind]
            for s in range(len(overlap)):
                tp1, fp1, fn1 = f_score(pred_data, acts, start, end, overlap[s], bg_action=pre_SIL)
                action_tp1[ind, s] += tp1
                action_fp1[ind, s] += fp1
                action_fn1[ind, s] += fn1

    for pred in test_files:
        video_id = int(pred.stem.split('/')[0])
        pred_lines = read_file(pred)
        if not pred_lines:
            print('no pred_line')
            continue
        pred_data = pred_lines[0].split()
        pred_num = [action_label[i] for i in pred_data]

        actions = test[test["video id"] == video_id]
        start = np.array(actions["start"])
        end = np.array(actions["end"])
        acts = np.array(actions["action label"])
        y_label = framewise_label(start, end, acts)

        for ind, label in enumerate(classes):
            total = 0
            correct = 0
            for i in range(min(len(y_label), len(pred_num))):
                if y_label[i] == label:
                    total += 1
                    if y_label[i] == pred_num[i]:
                        correct += 1
            action_total[ind] += total
            action_correct[ind] += correct

            pre_SIL = [x for i, x in enumerate(list(action_label.keys())) if i != ind]
            for s in range(len(overlap)):
                tp1, fp1, fn1 = f_score(pred_data, acts, start, end, overlap[s], bg_action=pre_SIL)
                action_tp1[ind, s] += tp1
                action_fp1[ind, s] += fp1
                action_fn1[ind, s] += fn1
    

    action_f1s = []
    for i in range(len(action_label)):
        action_acc[i] = 100 * (float(action_correct[i]) / action_total[i]) if action_total[i] > 0 else 0
        f1s = []
        for s in range(len(overlap)):
            precision = action_tp1[i, s] / float(action_tp1[i, s] + action_fp1[i, s])
            recall = action_tp1[i, s] / float(action_tp1[i, s] + action_fn1[i, s])
            f1 = 2. * (precision * recall) / (precision + recall) if (precision+recall) > 0 else 0
            f1 = f1 * 100
            f1s.append(f1)
        action_f1s.append(f1s)

    return action_acc, action_f1s

def action_eval(pred_path, label_path):
    pred_folder = Path(pred_path)
    try:
        pred_files = [file for file in pred_folder.iterdir() if file.suffix == '.txt']
    except FileNotFoundError as e:
        logger.error(f"folder not found: {e}")
        return

    label_data = read_file(label_path)
    overlap = [.1, .25, .5]
    pred_files.sort()
    classes = list(action_label.values())
    action_acc = [0] * len(action_label)
    action_total = [0] * len(action_label)
    action_correct = [0] * len(action_label)
    action_tp1 = np.zeros((len(action_label),3))
    action_fp1 = np.zeros((len(action_label),3))
    action_fn1 = np.zeros((len(action_label),3))

    for pred in pred_files:
        video_id = int(pred.stem.split('/')[0])
        pred_lines = read_file(pred)
        if not pred_lines:
            print('no pred_line')
            continue
        pred_data = pred_lines[0].split()
        pred_num = [action_label[i] for i in pred_data]

        actions = label_data[label_data["video id"] == video_id]
        start = np.array(actions["start"])
        end = np.array(actions["end"])
        acts = np.array(actions["action label"])
        y_label = framewise_label(start, end, acts)

        for ind, label in enumerate(classes):
            total = 0
            correct = 0
            for i in range(min(len(y_label), len(pred_num))):
                if y_label[i] == label:
                    total += 1
                    if y_label[i] == pred_num[i]:
                        correct += 1
            action_total[ind] += total
            action_correct[ind] += correct

            pre_SIL = [x for i, x in enumerate(list(action_label.keys())) if i != ind]
            for s in range(len(overlap)):
                tp1, fp1, fn1 = f_score(pred_data, acts, start, end, overlap[s], bg_action=pre_SIL)
                action_tp1[ind, s] += tp1
                action_fp1[ind, s] += fp1
                action_fn1[ind, s] += fn1

    action_f1s = []
    for i in range(len(action_label)):
        action_acc[i] = 100 * (float(action_correct[i]) / action_total[i]) if action_total[i] > 0 else 0
        f1s = []
        for s in range(len(overlap)):
            precision = action_tp1[i, s] / float(action_tp1[i, s] + action_fp1[i, s])
            recall = action_tp1[i, s] / float(action_tp1[i, s] + action_fn1[i, s])
            f1 = 2. * (precision * recall) / (precision + recall) if (precision+recall) > 0 else 0
            f1 = f1 * 100
            f1s.append(f1)
        action_f1s.append(f1s)

    return action_acc, action_f1s

def result_plt(label, id, paths):
    colors = color_list()
    predicts = []
    for pred in paths:
        for video in pred.iterdir():
            if video.suffix != ".txt":
                continue
            video_id = int(video.stem.split('/')[0])
            if video_id != id:
                continue
            pred_lines = read_file(video)
            if not pred_lines:
                print('no pred_line')
                continue
            pred_data = pred_lines[0].split()
            pred_num = [action_label[i] for i in pred_data]
            predicts.append(np.array(pred_num))

    label_data = read_file(label)
    actions = label_data[label_data["video id"] == id]
    start = np.array(actions["start"])
    end = np.array(actions["end"])
    acts = np.array(actions["action label"])
    y_label = framewise_label(start, end, acts)
    comparsion_chart(np.array(y_label), predicts, colors, Path(""))

def pre_visual(pred_path, quality_path):
    colors = color_list()
    pred_folder = Path(pred_path)
    #refine_dir = Path(pred_path) / "refinement"
    try:
        pred_files = [file for file in pred_folder.iterdir() if file.suffix == '.txt']
        #r_files = [file for file in refine_dir.iterdir() if file.suffix == '.txt']
    except FileNotFoundError as e:
        logger.error(f"folder not found: {e}")
        return
    for pred in pred_files:
        pred_lines = read_file(pred)
        if not pred_lines:
            print('no pred_line')
            continue
        pred_data = pred_lines[0].split()
        pred_num = [action_label[i] for i in pred_data]
        chart(None, pred_num, colors, quality_path)
    

def b_eval(pred_path, label_path, thr, tolerance):
    pred_folder = Path(pred_path)
    try:
        pred_files = [file for file in pred_folder.iterdir() if file.suffix == '.txt']
    except FileNotFoundError as e:
        logger.error(f"folder not found: {e}")
        return
    
    label_data = read_file(label_path)
    all_tp, all_fp, all_fn = 0, 0, 0
    correct = 0
    all_n_frame = 0


    for pred in pred_files:
        video_id = int(pred.stem.split('/')[0])
        pred_lines = read_file(pred)
        if not pred_lines:
            continue
        pred_data = pred_lines[0].split()
        pred_num = [float(i) for i in pred_data]
        pred_num = np.asarray(pred_num, dtype=np.float32)
        n_frames = pred_num.shape[0]
        pred_num = argrelmax(pred_num, threshold=thr)

        actions = label_data[label_data["video id"] == video_id]
        start = np.array(actions["start"])
        end = np.array(actions["end"])
        acts = np.array(actions["action label"])
        gt = boundary_label(start, end, acts)
        gt = np.asarray(gt, dtype=np.float32) 
        gt = argrelmax(gt, threshold=thr)

        tp = 0.0
        fp = 0.0
        fn = 0.0

        hits = np.zeros(len(gt))
        for i in range(len(pred_num)):
            dist = np.abs(np.array(gt) - pred_num[i])
            min_dist = np.min(dist)
            idx = np.argmin(dist)

            if min_dist <= tolerance and hits[idx] == 0:
                tp += 1
                hits[idx] = 1
            else:
                fp += 1

        fn = len(gt) - sum(hits)
        tn = n_frames - tp - fp - fn

        all_tp += tp
        all_fp += fp
        all_fn += fn
        all_n_frame += n_frames
        correct += tp + tn
    
    acc = 100 * correct / all_n_frame
    precision = all_tp / float(all_tp + all_fp)
    recall = all_tp / float(all_tp + all_fn)

    f1s = 2.0 * (precision * recall) / (precision + recall + 1e-7)
    f1s = np.nan_to_num(f1s) * 100
    logger.info("Boundary_Acc: %.4f, Boundary_F1: %.4f, Boundary_precision: %.4f, Boundary_recall: %.4f" %(acc, f1s, precision * 100, recall * 100))

    return acc, f1s, precision * 100, recall * 100
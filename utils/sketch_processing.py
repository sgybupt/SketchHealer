import numpy as np
import cv2
import random
# from rdp import rdp
# from interval import Interval, IntervalSet
import time
from torchvision.transforms import transforms
import torch

patch_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def canvas_size_google(sketch):
    """
    :param sketch: google sketch, quickDraw
    :return: int list,[x, y, h, w]
    """
    # get canvas size

    vertical_sum = np.cumsum(sketch[1:], axis=0)
    xmin, ymin, _ = np.min(vertical_sum, axis=0)
    xmax, ymax, _ = np.max(vertical_sum, axis=0)
    w = xmax - xmin
    h = ymax - ymin
    start_x = -xmin - sketch[0][0]
    start_y = -ymin - sketch[0][1]
    # sketch[0] = sketch[0] - sketch[0]
    return [int(start_x), int(start_y), int(h), int(w)]


def draw_three(sketch, window_name="google", padding=30,
               random_color=False, time=1, show=False, img_size=512):
    """
    :param sketches: google quickDraw, (n, 3)
    :param window_name: pass
    :param thickness: pass
    :return: None
    """
    # print("three ")
    # print(sketch)
    # print("-" * 70)
    thickness = int(img_size * 0.025)

    sketch = scale_sketch(sketch, (img_size, img_size))  # scale the sketch.
    [start_x, start_y, h, w] = canvas_size_google(sketch=sketch)
    start_x += thickness + 1
    start_y += thickness + 1
    canvas = np.ones((max(h, w) + 3 * (thickness + 1), max(h, w) + 3 * (thickness + 1), 3), dtype='uint8') * 255
    if random_color:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    else:
        color = (0, 0, 0)
    pen_now = np.array([start_x, start_y])
    first_zero = False
    for stroke in sketch:
        delta_x_y = stroke[0:0 + 2]
        state = stroke[2:]
        if first_zero:
            pen_now += delta_x_y
            first_zero = False
            continue
        cv2.line(canvas, tuple(pen_now), tuple(pen_now + delta_x_y), color, thickness=thickness)
        if int(state) == 1:  # next stroke
            first_zero = True
            if random_color:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            else:
                color = (0, 0, 0)
        pen_now += delta_x_y
    if show:
        key = cv2.waitKeyEx()
        if key == 27:  # esc
            cv2.destroyAllWindows()
            exit(0)
    # cv2.imwrite(f"./{window_name}.png", canvas)
    return cv2.resize(canvas, (img_size, img_size))


def make_graph(sketch, graph_num=30, graph_picture_size=128, padding=0, thickness=5,
               random_color=False, mask_prob=0.0, channel_3=False, save=False):
    """
    :param sketch: google quickDraw, (n, 3)
    :param random_color: single color for one stroke
    :param draw: if draw
    :param drawing: draw dynamic
    :param padding: if padding
    :param window_name: pass
    :param thickness: pass
    """
    tmp_img_size = 512
    thickness = int(tmp_img_size * 0.025)
    # preprocess
    sketch = scale_sketch(sketch, (tmp_img_size, tmp_img_size))  # scale the sketch.
    [start_x, start_y, h, w] = canvas_size_google(sketch=sketch)
    start_x += thickness + 1
    start_y += thickness + 1

    # graph (graph_num, 3, graph_size, graph_size)
    if channel_3:
        graphs = np.zeros((graph_num, graph_picture_size, graph_picture_size, 3), dtype='uint8')  # must uint8
    else:
        graphs = np.zeros((graph_num, graph_picture_size, graph_picture_size), dtype='uint8')  # must uint8
    # generate adj matrix
    adj_matrix = torch.eye(graph_num, dtype=torch.float) * 0.5  # (graph_num, graph_num)
    for index in range(graph_num):
        if index == 0:
            adj_matrix[0, 0] += 0.5
            continue
        # adj_matrix[index][(index + graph_num - 3) % graph_num] = 0.2
        adj_matrix[index][(index + graph_num - 2) % graph_num] = 0.2
        adj_matrix[index][(index + graph_num - 1) % graph_num] = 0.3
        adj_matrix[index][(index + graph_num) % graph_num] = 0.5
        adj_matrix[index][(index + graph_num + 1) % graph_num] = 0.3
        adj_matrix[index][(index + graph_num + 2) % graph_num] = 0.2
        # adj_matrix[index][(index + graph_num + 3) % graph_num] = 0.2
    adj_matrix[:, 0] += 0.5
    graph_count = 1
    # canvas (h, w, 3)
    if channel_3:
        canvas = np.zeros((max(h, w) + 2 * (thickness + 1), max(h, w) + 2 * (thickness + 1), 3), dtype='uint8') + 255
    else:
        canvas = np.zeros((max(h, w) + 2 * (thickness + 1), max(h, w) + 2 * (thickness + 1)), dtype='uint8')
    if random_color:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    else:
        color = (255, 255, 255)
    pen_now = np.array([start_x, start_y])
    first_zero = False

    # generate canvas.
    for index, stroke in enumerate(sketch):
        delta_x_y = stroke[0:0 + 2]
        state = stroke[2:]
        if first_zero:
            pen_now += delta_x_y
            first_zero = False
            continue
        cv2.line(canvas, tuple(pen_now), tuple(pen_now + delta_x_y), color, thickness=thickness)
        if int(state) != 0:  # next stroke
            first_zero = True
            if random_color:
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            else:
                color = (255, 255, 255)
        pen_now += delta_x_y
    # canvas_first = cv2.resize(canvas, (graph_picture_size, graph_picture_size))
    # graphs[0] = canvas_first

    if save:
        cv2.imwrite(f"./google.png", canvas)
    # generate patch pixel picture from canvas
    # make canvas larger, enlarge canvas 100 pixels boundary
    if channel_3:
        _h, _w, _c = canvas.shape  # (h, w, c)
        boundary_size = int(graph_picture_size * 1.5)
        top_bottom = np.zeros((boundary_size, _w, 3), dtype=canvas.dtype) + 255
        left_right = np.zeros((boundary_size * 2 + _h, boundary_size, 3), dtype=canvas.dtype) + 255
    else:
        _h, _w = canvas.shape  # (h, w, c)
        boundary_size = int(graph_picture_size * 1.5)
        top_bottom = np.zeros((boundary_size, _w), dtype=canvas.dtype)
        left_right = np.zeros((boundary_size * 2 + _h, boundary_size), dtype=canvas.dtype)
    canvas = np.concatenate((top_bottom, canvas, top_bottom), axis=0)
    canvas = np.concatenate((left_right, canvas, left_right), axis=1)
    # cv2.imwrite(f"./google_large.png", canvas)
    # processing.
    pen_now = np.array([start_x + boundary_size, start_y + boundary_size])
    first_zero = False
    mask_index_list = []  # to record with stroke is masked
    # generate patches.
    # strategies:
    # 1. get box at the head of one stroke
    # 2. in a long stroke, we get box in
    tmp_count = 0
    _move = graph_picture_size // 2
    for index, stroke in enumerate(sketch):
        delta_x_y = stroke[0:0 + 2]
        state = stroke[2:]
        if first_zero:
            pen_now += delta_x_y
            first_zero = False
            continue
        # cv2.line(canvas, tuple(pen_now), tuple(pen_now + delta_x_y), color=(255, 0, 0), thickness=thickness)
        if tmp_count % 4 == 0:
            tmpRec = canvas[pen_now[1] - _move:pen_now[1] + _move,
                     pen_now[0] - _move:pen_now[0] + _move]
            if graph_count > graph_num - 1:
                break
            if tmpRec.shape[0] != graph_picture_size or \
                    tmpRec.shape[1] != graph_picture_size:
                print(f'this sketch is broken: broken stroke: ', index)
                pass
            else:
                if np.random.random(1) < mask_prob:  # mask
                    canvas[pen_now[1] - _move:pen_now[1] + _move,
                    pen_now[0] - _move:pen_now[0] + _move] = 0  # mask this node.
                    mask_index_list.append(graph_count)  # add masked node into mask list
                    pass
                else:  # no mask
                    graphs[graph_count] = tmpRec
            graph_count += 1
            # cv2.line(canvas, tuple(pen_now), tuple(pen_now + np.array([1, 1])), color=(0, 0, 255), thickness=3)
            # cv2.rectangle(canvas,
            #               tuple(pen_now - np.array([graph_picture_size // 2, graph_picture_size // 2])),
            #               tuple(pen_now + np.array([graph_picture_size // 2, graph_picture_size // 2])),
            #               color=(255, 0, 0), thickness=1)
        tmp_count += 1
        if int(state) != 0:  # next stroke
            tmp_count = 0
            first_zero = True
        pen_now += delta_x_y
    if channel_3:
        graphs_tensor = torch.Tensor(graph_num, 3, graph_picture_size, graph_picture_size)
    else:
        graphs_tensor = torch.Tensor(graph_num, 1, graph_picture_size, graph_picture_size)
    # cv2.imwrite("./google_large_rec.png", canvas)
    # exit(0)
    masked_canvas = canvas[boundary_size: -boundary_size, boundary_size: -boundary_size]
    graphs[0] = cv2.resize(masked_canvas, (graph_picture_size, graph_picture_size))

    # mask block
    # remove rows and columns in mask list.
    for each_mask_index in mask_index_list:
        adj_matrix[each_mask_index, :] = 0
        adj_matrix[:, each_mask_index] = 0
    # remove reset blank nodes adj matrix.
    if graph_count < graph_num:
        adj_matrix[graph_count + 1:, :] = 0
        adj_matrix[:, graph_count + 1:] = 0

    for index in range(graph_num):
        graphs_tensor[index] = patch_trans(graphs[index])  # 此处变换的通道
        # cv2.imwrite(f"./{index}_trans.png", np.array(transforms.ToPILImage()(graphs_tensor[index])))
        # break
    # print(graphs_tensor[0], graphs_tensor[0].max(), graphs_tensor[0].std())
    # exit(0)

    return graphs_tensor, adj_matrix


def remove_white_space_image(img_np: np.ndarray):
    """
    :param img_np:
    :return:
    """
    if np.max(img_np) <= 1.0:  # 1.0 <= 1.0 True
        img_np = (img_np * 255).astype("uint8")
    else:
        img_np = img_np.astype("uint8")
    img_np_single = np.sum(img_np, axis=2)
    Y, X = np.where(img_np_single <= 760)  # max = 765,
    ymin, ymax, xmin, xmax = np.min(Y), np.max(Y), np.min(X), np.max(X)
    img_cropped = img_np[ymin:ymax, xmin:xmax, :]
    return img_cropped  # (h, w), img_cropped


def remove_white_space_sketch(sketch):
    """
    :param sketch:
    :return:
    """
    min_list = np.min(sketch, axis=0)
    sketch[:, :2] = sketch[:, :2] - np.array(min_list[:2])
    return sketch


def scale_sketch(sketch, size=(448, 448)):
    [_, _, h, w] = canvas_size_google(sketch)
    if h >= w:
        sketch_normalize = sketch / np.array([[h, h, 1]], dtype=np.float)
    else:
        sketch_normalize = sketch / np.array([[w, w, 1]], dtype=np.float)
    sketch_rescale = sketch_normalize * np.array([[size[0], size[1], 1]], dtype=np.float)
    return sketch_rescale.astype("int16")


if __name__ == '__main__':
    dataset_np = np.load("../dataset/alarm clock.npz", encoding='latin1', allow_pickle=True)
    print(list(dataset_np.keys()))
    t = time.time()
    graphs = []
    adjs = []
    npz_ = dataset_np['train']
    ranInt = random.randint(0, len(npz_))
    # ranInt = 30005
    print(f'random sample=', ranInt)
    for index, sample in enumerate(npz_):
        # sample = cat_np['train'][5845]  # 5845  # 5848
        # print(sample)
        gra, adj = make_graph(npz_[ranInt], graph_picture_size=64, graph_num=25 + 1,
                              mask_prob=0.1, random_color=False, channel_3=False, save=False)
        break
        # graphs.append(gra)
        # adjs.append(adj)
        # print(gra.shape, adj.shape)  # torch.Size([30, 3, 128, 128]) torch.Size([30, 30])
    # graphs_final = torch.from_numpy(np.stack(graphs, 0))
    # adj_final = torch.from_numpy(np.stack(adjs, 0))
    # print(graphs_final.shape, adj_final.shape)
    # print(f'process time: {(time.time() - t) / (graphs_final.shape[0])}')

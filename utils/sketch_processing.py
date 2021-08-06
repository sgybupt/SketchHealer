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
    读取quickDraw的画布大小及起始点
    :param sketch: google sketch, quickDraw
    :return: int list,[x, y, h, w]
    """
    # get canvas size

    vertical_sum = np.cumsum(sketch[1:], axis=0)  # 累加 排除第一笔未知的偏移量
    xmin, ymin, _ = np.min(vertical_sum, axis=0)
    xmax, ymax, _ = np.max(vertical_sum, axis=0)
    w = xmax - xmin
    h = ymax - ymin
    start_x = -xmin - sketch[0][0]  # 等效替换第一笔
    start_y = -ymin - sketch[0][1]
    # sketch[0] = sketch[0] - sketch[0]
    # 返回可能处理过的sketch
    return [int(start_x), int(start_y), int(h), int(w)]


def draw_three(sketch, window_name="google", padding=30,
               random_color=False, time=1, show=False, img_size=512):
    """
    此处主要包含画图部分，从canvas_size_google()获得画布的大小和起始点的位置，根据strokes来画
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
        if first_zero:  # 首个零是偏移量, 不画
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
               random_color=False, mask_prob=0.0, channel_3=False, save=""):
    tmp_img_size = 512
    thickness = int(tmp_img_size * 0.025)
    # preprocess
    sketch = scale_sketch(sketch, (tmp_img_size, tmp_img_size))  # scale the sketch.
    [start_x, start_y, h, w] = canvas_size_google(sketch=sketch)
    start_x += thickness + 1
    start_y += thickness + 1
    if channel_3:
        graphs = np.zeros((graph_num, graph_picture_size, graph_picture_size, 3), dtype='uint8')  # must uint8
    else:
        graphs = np.zeros((graph_num, graph_picture_size, graph_picture_size), dtype='uint8')  # must uint8

    # adjacent matrix
    adj_matrix = torch.zeros((graph_num, graph_num), dtype=torch.float)  # (graph_num, graph_num)

    # print(adj_matrix)
    graph_count = 0
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
        if first_zero:  # 首个零是偏移量, 不画
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

    # if save:
    #     cv2.imwrite(f"./google.png", canvas)

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

    # generate patches.
    # strategies:
    # 1. get box at the head of one stroke
    # 2. in a long stroke, we get box in

    batch_list = []  # [(index, point1, point2)]

    tmp_count = 0  # 每4笔 画一个框
    _move = graph_picture_size // 2
    for index, stroke in enumerate(sketch):
        delta_x_y = stroke[0:0 + 2]
        state = stroke[2:]
        if first_zero:  # 首个零是偏移量, 不画
            pen_now += delta_x_y
            first_zero = False
            continue
        # cv2.line(canvas, tuple(pen_now), tuple(pen_now + delta_x_y), color=(255, 0, 0), thickness=thickness)
        if tmp_count % 4 == 0:
            tmpRec = canvas[pen_now[1] - _move:pen_now[1] + _move,
                     pen_now[0] - _move:pen_now[0] + _move]
            if graph_count + 1 > graph_num - 1:  # 框足够了,break, 不足的已经补0了
                break
            if tmpRec.shape[0] != graph_picture_size or \
                    tmpRec.shape[1] != graph_picture_size:  # 出现问题的图片
                print(f'this sketch is broken: broken stroke: ', index)  # 忽略
                pass
            else:
                if np.random.uniform(0, 1) > mask_prob:  # 大于mask prob 则图像保留 否则是黑图

                    graphs[graph_count + 1] = tmpRec  # 第0张图是原图
                    batch_list.append((graph_count + 1, pen_now[1], pen_now[0]))
                else:
                    # adj_matrix[graph_count + 1, :] = 0
                    # adj_matrix[:, graph_count + 1] = 0
                    # 遮蔽对应位置, 防止以后有node 仍然利用了这个区域
                    canvas[pen_now[1] - _move:pen_now[1] + _move, pen_now[0] - _move:pen_now[0] + _move] = 0
            graph_count += 1  # 无论是否node是否被patch填充 都要加一

        tmp_count += 1
        if int(state) != 0:  # next stroke
            tmp_count = 0
            first_zero = True
        pen_now += delta_x_y
    if channel_3:
        graphs_tensor = torch.Tensor(graph_num, 3, graph_picture_size, graph_picture_size)
    else:
        graphs_tensor = torch.Tensor(graph_num, 1, graph_picture_size, graph_picture_size)
    # canvas_forshow = np.copy(canvas[boundary_size - thickness:-boundary_size + thickness,
    #                          boundary_size - thickness:-boundary_size + thickness])
    # canvas_forshow -= 255
    # canvas_forshow = -canvas_forshow

    # if save:
    #     cv2.imwrite(f"./final_pictures/mask/{mask_prob}.png", canvas_forshow)

    canvas_global = cv2.resize(canvas[boundary_size - thickness:-boundary_size + thickness,
                               boundary_size - thickness:-boundary_size + thickness],
                               (graph_picture_size, graph_picture_size))
    graphs[0] = canvas_global
    if save:
        aimPic = np.copy(canvas[boundary_size - thickness:-boundary_size + thickness,
                         boundary_size - thickness:-boundary_size + thickness])
        aimPic = aimPic - 255
        aimPic = - aimPic
        cv2.imwrite(save, aimPic)
    # 在最后生成 adjacent matrix
    # 计算所有节点的最大距离
    distence = 0
    distence_list = []  # reuse  [(distence, index1, index2)]
    for index, p1 in enumerate(batch_list):
        adj_matrix[p1[0]][p1[0]] = 0.5  # self weight
        adj_matrix[p1[0], 0] += 0.5  # global weight
        if index == len(batch_list) - 1:
            break
        for p2 in batch_list[index + 1:]:
            dis_tmp = np.sqrt((p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
            distence_list.append((dis_tmp, p1[0], p2[0]))
            distence = dis_tmp if dis_tmp > distence else distence

    near_threshold = 0.2
    distence_near = distence * near_threshold

    # nearby weight
    for i in distence_list:
        if i[0] <= distence_near:  # 小于threshold的点
            adj_matrix[i[1]][i[2]] = 0.3
            adj_matrix[i[2]][i[1]] = 0.3

    adj_matrix[0, 0] += 0.5  # global self

    if graph_count + 1 < graph_num:
        adj_matrix[graph_count + 1 + 1:, :] = 0
        adj_matrix[:, graph_count + 1 + 1:] = 0
    for index in range(graph_num):
        graphs_tensor[index] = patch_trans(graphs[index])  # 此处变换的通道
    return graphs_tensor, adj_matrix


def make_graph_(sketch, graph_num=30, graph_picture_size=128, padding=0, thickness=5,
                random_color=False, mask_prob=0.0, channel_3=False, save=False):
    """
    返回graphs, adj

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
            adj_matrix[0, 0] += 0.5  # 只跟自己有关
            continue
        # adj_matrix[index][(index + graph_num - 3) % graph_num] = 0.2
        adj_matrix[index][(index + graph_num - 2) % graph_num] = 0.2
        adj_matrix[index][(index + graph_num - 1) % graph_num] = 0.3
        adj_matrix[index][(index + graph_num) % graph_num] = 0.5
        adj_matrix[index][(index + graph_num + 1) % graph_num] = 0.3
        adj_matrix[index][(index + graph_num + 2) % graph_num] = 0.2
        # adj_matrix[index][(index + graph_num + 3) % graph_num] = 0.2
    adj_matrix[:, 0] += 0.5  # 补全 全局的权重
    # print(adj_matrix)
    graph_count = 0
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
        if first_zero:  # 首个零是偏移量, 不画
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
    canvas_first = cv2.resize(canvas, (graph_picture_size, graph_picture_size))
    graphs[0] = canvas_first

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

    # generate patches.
    # strategies:
    # 1. get box at the head of one stroke
    # 2. in a long stroke, we get box in
    tmp_count = 0  # 每4笔 画一个框
    _move = graph_picture_size // 2
    for index, stroke in enumerate(sketch):
        delta_x_y = stroke[0:0 + 2]
        state = stroke[2:]
        if first_zero:  # 首个零是偏移量, 不画
            pen_now += delta_x_y
            first_zero = False
            continue
        # cv2.line(canvas, tuple(pen_now), tuple(pen_now + delta_x_y), color=(255, 0, 0), thickness=thickness)
        if tmp_count % 4 == 0:
            tmpRec = canvas[pen_now[1] - _move:pen_now[1] + _move,
                     pen_now[0] - _move:pen_now[0] + _move]
            if graph_count + 1 > graph_num - 1:  # 框足够了,break, 不足的已经补0了
                break
            if tmpRec.shape[0] != graph_picture_size or \
                    tmpRec.shape[1] != graph_picture_size:  # 出现问题的图片
                print(f'this sketch is broken: broken stroke: ', index)  # 忽略
                pass
            else:
                graphs[graph_count + 1] = tmpRec  # 第0张图是原图
																 						
            graph_count += 1
            # 保存框
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

    for index in range(graph_num):
        graphs_tensor[index] = patch_trans(graphs[index])  # 此处变换的通道
        # cv2.imwrite(f"./{index}_trans.png", np.array(transforms.ToPILImage()(graphs_tensor[index])))
        # break
    # mask block
    mask_list = [x for x in range(graph_count)]
    mask_list.remove(0)  # remove global, prevent be masked
    mask_number = int(mask_prob * graph_count)
    mask_index_list = random.sample(mask_list, mask_number)
    for mask_index in mask_index_list:
        graphs[mask_index, :] = 0
        adj_matrix[mask_index, :] = 0
        adj_matrix[:, mask_index] = 0
    if graph_count + 1 < graph_num:
        adj_matrix[graph_count + 1 + 1:, :] = 0
        adj_matrix[:, graph_count + 1 + 1:] = 0
    # print(graphs_tensor[0], graphs_tensor[0].max(), graphs_tensor[0].std())
    # exit(0)

    return graphs_tensor, adj_matrix


def remove_white_space_image(img_np: np.ndarray):
    """
    获取白底图片中, 物体的bbox; 此处白底必须是纯白色.
    其中, 白底有两种表示方法, 分别是 1.0 以及 255; 在开始时进行检查并且匹配
    对最大值为255的图片进行操作.
    三通道的图无法直接使用255进行操作, 为了减小计算, 直接将三通道相加, 值为255*3的pix 认为是白底.
    :param img_np:
    :return:
    """
    if np.max(img_np) <= 1.0:  # 1.0 <= 1.0 True
        img_np = (img_np * 255).astype("uint8")
    else:
        img_np = img_np.astype("uint8")
    img_np_single = np.sum(img_np, axis=2)
    Y, X = np.where(img_np_single <= 760)  # max = 765, 留有一些余地
    ymin, ymax, xmin, xmax = np.min(Y), np.max(Y), np.min(X), np.max(X)
    img_cropped = img_np[ymin:ymax, xmin:xmax, :]
    return img_cropped  # (h, w), img_cropped


def remove_white_space_sketch(sketch):
    """
    删除留白
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
    import glob
    import os

    prob = 0.3
    for each_np_path in glob.glob("../dataset/*.npz"):
        catname = each_np_path.split("/")[-1].split(".")[0]
        os.makedirs(f"/root/human-study/human/{prob}/{catname}", exist_ok=True)
        dataset_np = np.load(each_np_path, encoding='latin1', allow_pickle=True)
			 
        npz_ = dataset_np['test']
										 
					
									
        for index, sample in enumerate(npz_):
														
					   
            gra, adj = make_graph(sample, graph_picture_size=64,
                                  mask_prob=prob, random_color=False, channel_3=False,
                                  save=f"/root/human-study/human/{prob}/{catname}/{index}.jpg")
            print(index)
						 

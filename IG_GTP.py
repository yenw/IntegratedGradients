#!/usr/bin/env python
# encoding=utf8

import tensorflow as tf
import numpy as np
import sys
import time
import math

next_black = 1
ko_x = -2
ko_y = -2
dead_count = 0
dead_try = 0
board = [[0 for i in range(19)] for j in range(19)]
history = [[0.0 for i in range(361)] for c in range(16)]

def gtp_log(str):
    sys.stderr.write("%s\n" % (str))
    sys.stderr.flush()

def get_sess():
    saver = tf.train.import_meta_graph("meta_graph")
    ckpt = tf.train.get_checkpoint_state("./")
    sess = tf.InteractiveSession()
    saver.restore(sess, ckpt.model_checkpoint_path)
    graph = sess.graph

    return sess, graph

sess, graph = get_sess()

def try_eat(x, y, foe_color, update = True):
    global board, history, dead_count, dead_try

    new_mask = 1
    mask = [[0 for i in range(19)] for j in range(19)]
    stack_x = [0 for i in range(361)]
    stack_y = [0 for i in range(361)]
    stack_i = 0

    stack_x[stack_i] = x
    stack_y[stack_i] = y
    stack_i += 1
    dead = True
    while stack_i != 0:
        stack_i -= 1
        dx = stack_x[stack_i]
        dy = stack_y[stack_i]

        if mask[dx][dy] == new_mask:
            continue

        mask[dx][dy] = new_mask
        if dx > 0:
            if board[dx - 1][dy] == 0:
                dead = False
                break
            elif board[dx - 1][dy] == foe_color and mask[dx - 1][dy] != new_mask:
                stack_x[stack_i] = dx - 1
                stack_y[stack_i] = dy
                stack_i += 1

        if dx < 18:
            if board[dx + 1][dy] == 0:
                dead = False
                break
            elif board[dx + 1][dy] == foe_color and mask[dx + 1][dy] != new_mask:
                stack_x[stack_i] = dx + 1
                stack_y[stack_i] = dy
                stack_i += 1

        if dy > 0:
            if board[dx][dy - 1] == 0:
                dead = False
                break
            elif board[dx][dy - 1] == foe_color and mask[dx][dy - 1] != new_mask:
                stack_x[stack_i] = dx
                stack_y[stack_i] = dy - 1
                stack_i += 1

        if dy < 18:
            if board[dx][dy + 1] == 0:
                dead = False
                break
            elif board[dx][dy + 1] == foe_color and mask[dx][dy + 1] != new_mask:
                stack_x[stack_i] = dx
                stack_y[stack_i] = dy + 1
                stack_i += 1

    if dead and update:
        for y in range(19):
            for x in range(19):
                if mask[x][y] == new_mask and board[x][y] == foe_color:
                    board[x][y] = 0
                    dead_count += 1
    else:
        dead_try = 0
        for y in range(19):
            for x in range(19):
                if mask[x][y] == new_mask and board[x][y] == foe_color:
                    dead_try += 1
    return dead

def legal(x, y, self_color, foe_color):
    global board, next_black, ko_x, ko_y

    if x == ko_x and y == ko_y:
        return 0

    if board[x][y] != 0:
        return 0

    if x > 0 and board[x - 1][y] == 0:
        return 1

    if x < 18 and board[x + 1][y] == 0:
        return 1

    if y > 0 and board[x][y - 1] == 0:
        return 1

    if y < 18 and board[x][y + 1] == 0:
        return 1

    if x > 0 and board[x - 1][y] == foe_color:
        board[x][y] = 3
        if try_eat(x - 1, y, foe_color, False):
            board[x][y] = 0
            return 1
        else:
            board[x][y] = 0

    if x < 18 and board[x + 1][y] == foe_color:
        board[x][y] = 3
        if try_eat(x + 1, y, foe_color, False):
            board[x][y] = 0
            return 1
        else:
            board[x][y] = 0

    if y > 0 and board[x][y - 1] == foe_color:
        board[x][y] = 3
        if try_eat(x, y - 1, foe_color, False):
            board[x][y] = 0
            return 1
        else:
            board[x][y] = 0

    if y < 18 and board[x][y + 1] == foe_color:
        board[x][y] = 3
        if try_eat(x, y + 1, foe_color, False):
            board[x][y] = 0
            return 1
        else:
            board[x][y] = 0

    board[x][y] = self_color
    dead = try_eat(x, y, self_color, False)
    board[x][y] = 0
    if dead:
        return 0

    return 1

def get_planes(rate = 1.0):
    global history, next_black
    color_val = float(next_black == 1)
    planes = []
    for i in range(361):
        for k in range(16):
            planes.append(history[k][i] * rate)
        planes.append(color_val)

    return planes

def get_planes_black(rate = 1.0):
    global history, next_black
    color_val = float(next_black == 1)
    planes = []
    for i in range(361):
        for k in range(16):
            if ((k % 2) == 0):
                planes.append(history[k][i] * rate)
            else:
                planes.append(history[k][i] * 1.0)
        planes.append(color_val)

    return planes

def get_planes_white(rate = 1.0):
    global history, next_black
    color_val = float(next_black == 1)
    planes = []
    for i in range(361):
        for k in range(16):
            if ((k % 2) == 1):
                planes.append(history[k][i] * rate)
            else:
                planes.append(history[k][i] * 1.0)
        planes.append(color_val)

    return planes

def gen_move():
    global sess, graph
    planes = get_planes()
    policy = sess.run(graph.get_tensor_by_name('policy:0'), feed_dict={graph.get_tensor_by_name('Cast:0'): [planes]})
    policy = policy[0]
    max_idx = 361
    max_policy = policy[max_idx]
    for i in range(361):    
        if policy[i] > max_policy:
            max_idx = i
            max_policy = policy[max_idx]

    if max_idx == 361:
        return -1, -1
    else:
        return int(max_idx % 19), int(max_idx / 19)

def update_history():
    global board, history
    for i in range(7):
        history[i + 2] = history[i]
        history[i + 3] = history[i + 1]

    history[0] = [0 for i in range(361)]
    history[1] = [0 for i in range(361)]
    for y in range(19):
        for x in range(19):
            if board[x][y] == 1:
                history[0][x + y * 19] = 1.0
            elif board[x][y] == 2:
                history[1][x + y * 19] = 1.0

def play_pass():
    global ko_x, ko_y, dead_count

    ko_x = -2
    ko_y = -2
    dead_count = 0
    update_history()

def play(x, y):
    global board, history, next_black, ko_x, ko_y, dead_count
    foe_color = 0
    if next_black == 1:
        foe_color = 2
        board[x][y] = 1
        next_black = 2
    else:
        foe_color = 1
        board[x][y] = 2
        next_black = 1

    dead_count = 0
    if x > 0 and board[x - 1][y] == foe_color:
        try_eat(x - 1, y, foe_color)

    if x < 18 and board[x + 1][y] == foe_color:
        try_eat(x + 1, y, foe_color)

    if y > 0 and board[x][y - 1] == foe_color:
        try_eat(x, y - 1, foe_color)

    if y < 18 and board[x][y + 1] == foe_color:
        try_eat(x, y + 1, foe_color)

    if dead_count == 1:
        empty = 0
        if x > 0 and board[x - 1][y] == 0:
            empty += 1
            ko_x = x - 1
            ko_y = y

        if x < 18 and board[x + 1][y] == 0:
            empty += 1
            ko_x = x + 1
            ko_y = y

        if y > 0 and board[x][y - 1] == 0:
            empty += 1
            ko_x = x
            ko_y = y - 1

        if y < 18 and board[x][y + 1] == 0:
            empty += 1
            ko_x = x
            ko_y = y + 1

        if empty > 1:
            ko_x = -2
            ko_y = -2
    else:
        ko_x = -2
        ko_y = -2

    update_history()

def print_gradient(g):
    channels = len(g)/361
    for c in range(channels):
        s = sum([g[c + i * channels] for i in range(361)])
        print "channel %d: --> sum() = %s" % (c, s)
        for y in range(19):
            for x in range(19):
                print "%+.6f" % (g[c + (x + y * 19) * channels]),

            print

    print "sum(channel):"
    for y in range(19):
        for x in range(19):
            p = (x + y * 19) * channels
            s = sum([g[c + p] for c in range(channels - 1)])
            print "%+.6f" % s,

        print

def get_sum_gradient(g, sum_channles=16):
    channels = len(g)/361
    s = []
    for y in range(19):
        for x in range(19):
            p = (x + y * 19) * channels
            s.append(sum([g[c + p] for c in range(sum_channles)]))
    
    return s

def HSV2RGB(H, S, V):
    hue = H * 6.0;
    i = int(math.floor(hue));
    f = hue - i;
    p = V * (1.0 - S);

    R = 0.0
    G = 0.0
    B = 0.0

    if i == 0:
        R = V;
        G = V * (1.0 - S * (1.0 - f));
        B = p;
    elif i == 1:
        R = V * (1.0 - S * f);
        G = V;
        B = p;
    elif i == 2:
        R = p;
        G = V;
        B = V * (1.0 - S * (1.0 - f));
    elif i == 3:
        R = p;
        G = V * (1.0 - S * f);
        B = V;
    elif i == 4:
        R = V * (1.0 - S * (1.0 - f));
        G = p;
        B = V;
    else:
        R = V;
        G = p;
        B = V * (1.0 - S * f);

    return R, G, B

def double2hex(i):
    return "%02x" % int(i * 255)

def gogui_gfx_raw(gfx_board, reverse = False):
    gfx_gogui = [gfx_board[i] for i in range(len(gfx_board))]
    gfx_min = min(gfx_gogui)
    gfx_max = max(gfx_gogui)
    gfx_range = gfx_max - gfx_min
    if reverse:
        gfx_min = max(gfx_gogui)
        gfx_max = min(gfx_gogui)
        gfx_range = gfx_max - gfx_min

    gfx_str = ""
    H1 = 210.0 / 360;
    for y in range(19):
        for x in range(19):
            p = x + y * 19
            value = gfx_board[p]
            value_gogui = gfx_gogui[p]
            coord = chr(65 + x) + "%d" % (19 - y)
            if x >= 8:
                coord = chr(65 + x + 1) + "%d" % (19 - y)

            if gfx_range == 0:
                continue

            if value == 0:
                continue

            gfx_str += "LABEL %s %.2f\n" % (coord, value_gogui);
            H = 1.0 - float(value_gogui - gfx_min) / (gfx_range)
            R, G, B = HSV2RGB(H * H1, 1.0, 1.0)
            info = " #" + double2hex(R) + double2hex(G) + double2hex(B) + " " + coord;
            gfx_str += "COLOR" + info + "\n";

    return gfx_str

def gogui_gfx(gfx_board):
    gfx_gogui = [int(gfx_board[i] * 1000) for i in range(len(gfx_board))]
    gfx_min = min(gfx_gogui)
    gfx_max = max(gfx_gogui)
    gfx_range = gfx_max - gfx_min
    gfx_str = ""
    H1 = 210.0 / 360;
    for y in range(19):
        for x in range(19):
            p = x + y * 19
            value = gfx_board[p]
            value_gogui = gfx_gogui[p]
            coord = chr(65 + x) + "%d" % (19 - y)
            if x >= 8:
                coord = chr(65 + x + 1) + "%d" % (19 - y)

            if gfx_range == 0:
                continue

            if value == 0:
                continue

            gfx_str += "LABEL " + coord + " " + str(value_gogui) + "\n";
            H = 1.0 - float(value_gogui - gfx_min) / (gfx_range)
            R, G, B = HSV2RGB(H * H1, 1.0, 1.0)
            info = " #" + double2hex(R) + double2hex(G) + double2hex(B) + " " + coord;
            gfx_str += "COLOR" + info + "\n";

    return gfx_str

def integrated_gradients_gogui(g, sum_channles):
    ig_board = get_sum_gradient(g, sum_channles)
    return gogui_gfx(ig_board)

def integrated_gradients_impl(sum_channles, f, multi = True):
    global sess, graph, next_black

    step = 10
    ig = []
    for i in range(step):
        rate = float(i) / float(step - 1)
        gtp_log("%s / %s" % (i + 1, step))
        planes = f(rate)
        input_tensor = graph.get_tensor_by_name('Cast:0')
        output_tensor = graph.get_tensor_by_name('value:0')
        gradients_node = tf.gradients(ys=output_tensor, xs=input_tensor)
        grads = sess.run([gradients_node], feed_dict={graph.get_tensor_by_name('Cast:0'): [planes]})
        ig.append(grads[0][0][0].tolist())

    next_black_backup = next_black
    next_black = 2
    planes = f()
    next_black = next_black_backup
    output = []

    for i in range(len(planes)):
        s = 0.0
        for _step in range(step):
            s += ig[_step][i]

        if multi:
            s = planes[i] * s / float(step)
        else:
            s = s / float(step)
        output.append(s)

    return output

def integrated_gradients(sum_channles, f, f2=0):
    output = integrated_gradients_impl(sum_channles, f)
    if f2 != 0:
        gtp_log("second:")
        output_2 = integrated_gradients_impl(sum_channles, f2)
        output = [output[i] + output_2[i] for i in range(len(output))]

    #print_gradient(output)
    return integrated_gradients_gogui(output, sum_channles)

def integrated_gradients2(sum_channles, f):
    output = integrated_gradients_impl(sum_channles, f, multi = False)
    return integrated_gradients_gogui(output, sum_channles)

def gradients_impl(sum_channles, f):
    global sess, graph, next_black
    g = []
    planes = f()
    input_tensor = graph.get_tensor_by_name('Cast:0')
    output_tensor = graph.get_tensor_by_name('value:0')
    gradients_node = tf.gradients(ys=output_tensor, xs=input_tensor)
    grads = sess.run([gradients_node], feed_dict={graph.get_tensor_by_name('Cast:0'): [planes]})
    return grads[0][0][0].tolist()

def gradients(sum_channles, f):
    output = gradients_impl(sum_channles, f)
    return integrated_gradients_gogui(output, sum_channles)

def get_planes_without_point(point, color):
    global history, next_black
    color_val = float(next_black == 1)
    planes = []
    for i in range(361):
        for k in range(16):
            planes.append(history[k][i])
        planes.append(color_val)

    for k in range(8):
        k2 = k * 2 + color
        planes[point * 17 + k2] = 0.0
    return planes

def winrate_impl(planes):
    global sess, graph
    winrate = sess.run(graph.get_tensor_by_name('value:0'), feed_dict={graph.get_tensor_by_name('Cast:0'): [planes]})
    return (winrate[0] + 1.0) / 2.0

def winrate_board(show_black = True, show_white = True):
    base_winrate = winrate_impl(get_planes())
    gtp_log("winrate: " + str(base_winrate))

    step = 0
    for y in range(19):
        for x in range(19):
            if board[x][y] == 1 and show_black:
                step += 1
            elif board[x][y] == 2 and show_white:
                step += 1

    output = [0 for i in range(361)]
    i = 0
    for y in range(19):
        for x in range(19):
            if board[x][y] != 0:
                p = x + y * 19
                if board[x][y] == 1 and show_black:
                    i += 1
                    gtp_log("%s / %s" % (i, step))
                    
                    black_planes = get_planes_without_point(p, 0)
                    black_winrate = winrate_impl(black_planes)
                    output[p] = base_winrate - black_winrate
                elif board[x][y] == 2 and show_white:
                    i += 1
                    gtp_log("%s / %s" % (i, step))
                    
                    white_planes = get_planes_without_point(p, 1)
                    white_winrate = winrate_impl(white_planes)
                    output[p] = base_winrate - white_winrate

    if show_white and (not show_black):
        return gogui_gfx_raw(output, reverse = True)
    else:
        return gogui_gfx_raw(output)

def show():
    global board
    print " " * 2,
    for y in range(19):
        print y % 10,

    print
    for y in range(19):
        if y < 10:
            print " %d" % y,
        else:
            print "%d" % y,
        for x in range(19):
            if board[x][y] == 1:
                print "X",
            elif board[x][y] == 2:
                print "O",
            else:
                print ".",
        if y < 10:
            print " %d" % y
        else:
            print "%d" % y

    print " " * 2,
    for y in range(19):
        print y % 10,

    print

def clear_board():
    global board, history, next_black, foe_pass, ko_x, ko_y, dead_count, foe_pass
    next_black = 1
    ko_x = -2
    ko_y = -2
    dead_count = 0
    foe_pass = 0
    board = [[0 for i in range(19)] for j in range(19)]
    history = [[0.0 for i in range(361)] for c in range(16)]

def gtp_genmove():
    x, y = gen_move()
    if x == -1:
        return "PASS"

    play(x, y)
    if x > 7:
        x += 1

    coord = chr(65 + x) + "%d" % (19 - y)
    return coord

def gtp_print(str):
    sys.stdout.write("= %s\n\n" % (str))
    sys.stdout.flush()

def gtp_mode():
    global sgf, board, next_black, foe_pass
    gtp_cmdmands = [
    "name",
    "version",
    "protocol_version",
    "list_commands",
    "boardsize",
    "clear_board",
    "genmove",
    "genmove_black",
    "genmove_white",
    "play",
    "quit",
    "show",
    "gogui_analyze_commands"
    ]
    while True:
        #time.sleep(0.5)
        cmd = sys.stdin.readline().strip(' \n')
        if cmd == "name":
            gtp_print("PhoenixGoIG")
        elif cmd == "version":
            gtp_print("1.0")
        elif cmd == "protocol_version":
            gtp_print("2")
        elif cmd == "list_commands":
            gtp_print("\n".join(gtp_cmdmands))
        elif cmd == "boardsize 19":
            gtp_print("")
        elif cmd == "clear_board":
            clear_board()
            gtp_print("")
        elif cmd == "genmove B" or cmd == "genmove b" or cmd == "genmove_black":
            next_black = 1
            coord = gtp_genmove()
            gtp_print(coord)
        elif cmd == "genmove W" or cmd == "genmove w" or cmd == "genmove_white":
            next_black = 2
            coord = gtp_genmove()
            gtp_print(coord)
        elif cmd == "quit":
            return
        elif cmd == "show":
            show()
            gtp_print("")
        elif cmd == "gogui_analyze_commands":
            ana = "gfx/Null/null"
            ana += "\ngfx/Winrate_B/winrate_board_b"
            ana += "\ngfx/Winrate_W/winrate_board_w"
            ana += "\ngfx/Winrate_B+W/winrate_board_bw"
            ana += "\ngfx/Gradient_16/gradient_16"
            ana += "\ngfx/Gradient_17/gradient_17"
            ana += "\ngfx/IG_Src_2/ig1_2"
            ana += "\ngfx/IG_Src_16/ig1_16"
            ana += "\ngfx/IG_Black_16/ig2_1_16"
            ana += "\ngfx/IG_White_16/ig2_2_16"
            ana += "\ngfx/IG_B+W_16/ig2_3_16"
            ana += "\ngfx/G_2/ig3_2"
            ana += "\ngfx/G_16/ig3_16"
            ana += "\ngfx/G_17/ig3_17"
            gtp_print(ana)
        elif cmd == "null":
            gtp_print("")
        elif cmd == "winrate_board_bw":
            wr_str = winrate_board(show_black = True, show_white = True)
            gtp_print(wr_str)
        elif cmd == "winrate_board_b":
            wr_str = winrate_board(show_black = True, show_white = False)
            gtp_print(wr_str)
        elif cmd == "winrate_board_w":
            wr_str = winrate_board(show_black = False, show_white = True)
            gtp_print(wr_str)
        elif cmd == "gradient_16":
            ig_str = gradients(16, get_planes)
            gtp_print(ig_str)
        elif cmd == "gradient_17":
            ig_str = gradients(17, get_planes)
            gtp_print(ig_str)
        elif cmd == "ig1_2":
            ig_str = integrated_gradients(2, get_planes)
            gtp_print(ig_str)
        elif cmd == "ig1_16":
            ig_str = integrated_gradients(16, get_planes)
            gtp_print(ig_str)
        elif cmd == "ig2_1_16":
            ig_str = integrated_gradients(16, get_planes_black)
            gtp_print(ig_str)
        elif cmd == "ig2_2_16":
            ig_str = integrated_gradients(16, get_planes_white)
            gtp_print(ig_str)
        elif cmd == "ig2_3_16":
            ig_str = integrated_gradients(16, get_planes_black, get_planes_white)
            gtp_print(ig_str)
        elif cmd == "ig3_2":
            ig_str = integrated_gradients2(2, get_planes)
            gtp_print(ig_str)
        elif cmd == "ig3_16":
            ig_str = integrated_gradients2(16, get_planes)
            gtp_print(ig_str)
        elif cmd == "ig3_17":
            ig_str = integrated_gradients2(17, get_planes)
            gtp_print(ig_str)
        else:
            cmds = cmd.split(" ")
            if cmds[0] == "play":
                if cmds[1] == "B" or cmds[1] == "b":
                    next_black = 1
                    if cmds[2] == "pass" or cmds[2] == "Pass" or cmds[2] == "PASS":
                        play_pass()
                        foe_pass += 1
                    else:
                        x = ord(cmds[2][0]) - 65
                        if x > 19:
                            x -= 32

                        if x > 7:
                            x -= 1

                        y = 19 - int(cmds[2][1:])
                        play(x, y)
                        foe_pass = 0
                    #show()
                    gtp_print("")
                elif cmds[1] == "W" or cmds[1] == "w":
                    next_black = 2
                    if cmds[2] == "pass" or cmds[2] == "Pass" or cmds[2] == "PASS":
                        play_pass()
                        foe_pass += 1
                    else:
                        x = ord(cmds[2][0]) - 65
                        if x > 19:
                            x -= 32

                        if x > 7:
                            x -= 1

                        y = 19 - int(cmds[2][1:])
                        play(x, y)
                        foe_pass = 0
                    #show()
                    gtp_print("")
                else:
                    gtp_print("")
            else:
                gtp_print("")

gtp_mode()

################################################################################

'''
def get_x():
    x = []
    for i in range(361):
        for k in range(16):
            x.append(False)
        x.append(True)
    return x

def get_x_float(rate=1.0):
    x = []
    for i in range(361):
        for k in range(16):
            x.append(0.0)
        x.append(0.0)

    #[B_t, W_t, B_{t-1}, W_{t-1}, C] -> 8 * 2 + 1
    #x[0 + (2 + 3 * 19) * 17] = rate
    #x[1 + (3 + 16 * 19) * 17] = rate
    #x[2 + (2 + 3 * 19) * 17] = rate
    x[0 + (2 + 3 * 19) * 17] = rate
    x[0 + (1 + 1 * 19) * 17] = rate
    x[1 + (3 + 16 * 19) * 17] = rate
    x[2 + (2 + 3 * 19) * 17] = rate
    x[3 + (3 + 16 * 19) * 17] = rate
    x[4 + (2 + 3 * 19) * 17] = rate
    return x

def print_policy1(p):
    xp = []
    for y in range(19):
        tmp = []
        for x in range(19):
            print p[x + y * 19],

        print

    print p[361]

def print_policy_gogui(p):
    p_sum = sum(p)
    p_max = max(p)
    p_min = min(p)
    p_range = p_max - p_min

    for y in range(19):
        for x in range(19):
            print ("%4d"%((p[x + y * 19]) * 1000.0 / p_sum)),

        print

    print ("%4d"%((p[361]) * 1000.0 / p_sum))

def print_layer(l):
    for c in range(len(l[0][0])):
    #for c in range(1):
        print "channel %d: " % c
        for y in range(19):
            for x in range(19):
                print l[y][x][c],
                #print ("%+.6f"%l[y][x][c]),

            print

def print_policy(p):
    print "map %d: " % 1
    for y in range(19):
        for x in range(19):
            print p[x + y * 19],
            #print ("%+.6f"%l[y][x][c]),

        print
    print p[361]

def print_value(v):
    print "map %d: " % 1
    for y in range(16):
        for x in range(16):
            print v[x + y * 16],
            #print ("%+.6f"%l[y][x][c]),

        print

def print_gradient(g):
    channels = len(g)/361
    for c in range(channels):
        s = sum([g[c + i * channels] for i in range(361)])
        print "channel %d: --> sum() = %s" % (c, s)
        for y in range(19):
            for x in range(19):
                print "%+.6f" % (g[c + (x + y * 19) * channels]),

            print

    print "sum(channel):"
    for y in range(19):
        for x in range(19):
            p = (x + y * 19) * channels
            s = sum([g[c + p] for c in range(channels - 1)])
            print "%+.6f" % s,

        print

def print_gradient_gogui(g):
    channels = len(g)/361
    for c in range(channels):
        board = []
        for y in range(19):
            for x in range(19):
                board.append(g[c + (x + y * 19) * channels])

        print "channel %d: " % c
        p_sum = sum(board)
        p_max = max(board)
        p_min = min(board)
        p_range = p_max - p_min
        p_sum2 = p_sum - p_min * 361

        for y in range(19):
            for x in range(19):
                if p_sum2 != 0:
                    print "%4d" % int((board[x + y * 19] - p_min) * 1000.0 / p_sum2),
                else:
                    print "%4d" % (0),

            print

def print_ig(ig):
    print "ig: "
    for y in range(19):
        for x in range(19):
            print "%+.6f" % ig[x + y * 19],

        print

def show_pv():
    saver = tf.train.import_meta_graph("meta_graph")
    ckpt = tf.train.get_checkpoint_state("./")
    sess = tf.InteractiveSession()
    saver.restore(sess, ckpt.model_checkpoint_path)
    g = sess.graph

    x = get_x()
    x_float = get_x_float()

    #for i in range(20):
    #    rate = float(i) / float(19)
    #    print rate
    #    x_float = get_x_float(rate)
    #    p2 = sess.run(g.get_tensor_by_name('policy:0'), feed_dict={g.get_tensor_by_name('Cast:0'): [x_float]})
    #    v2 = sess.run(g.get_tensor_by_name('value:0'), feed_dict={g.get_tensor_by_name('Cast:0'): [x_float]})
    #    print "value: %s" % ((v2[0] + 1.0) / 2.0)
    #    print "policy sum: %s" % sum(p2[0])
    #    print "policy:"
    #    print_policy_gogui(p2[0])

    #p1 = sess.run(g.get_tensor_by_name('policy:0'), feed_dict={g.get_tensor_by_name('inputs:0'): [x]})
    #v1 = sess.run(g.get_tensor_by_name('value:0'), feed_dict={g.get_tensor_by_name('inputs:0'): [x]})

    #print "value: %s" % ((v1[0] + 1.0) / 2.0)
    #print "policy sum: %s" % sum(p1[0])
    #print "policy:"
    #print_policy_gogui(p1[0])

    step = 20
    ig = []
    for i in range(step):
        rate = float(i) / float(step - 1)
        print "%s / %s" % (i + 1, step)
        x_float = get_x_float(rate)
        input_tensor = g.get_tensor_by_name('Cast:0')
        output_tensor = g.get_tensor_by_name('value:0')
        gradients_node = tf.gradients(ys=output_tensor, xs=input_tensor)
        grads = sess.run([gradients_node], feed_dict={g.get_tensor_by_name('Cast:0'): [x_float]})
        #print_gradient(grads[0][0][0])
        ig.append(grads[0][0][0].tolist())

        p2 = sess.run(g.get_tensor_by_name('policy:0'), feed_dict={g.get_tensor_by_name('Cast:0'): [x_float]})
        v2 = sess.run(g.get_tensor_by_name('value:0'), feed_dict={g.get_tensor_by_name('Cast:0'): [x_float]})
        print "value: %s" % ((v2[0] + 1.0) / 2.0)
        print "policy sum: %s" % sum(p2[0])
        print "policy:"
        print_policy_gogui(p2[0])

    x_float = get_x_float(1.0)
    output = []

    for i in range(len(x_float)):
        s = 0.0
        for _step in range(step):
            s += ig[_step][i]

        s = x_float[i] * s / float(step)
        output.append(s)

    print_gradient(output)

#show_pv()
'''

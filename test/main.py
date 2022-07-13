

def print_hi(name):
    print(f'Hi, {name}')


chinese_num_dict = {'0': '零', '1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九',
                    '.': '点'}



def to_zh(num):
    num_int = int(num * 100)

    chinese_num = ""
    num_int = str(num_int)
    if len(num_int) == 4:
        if num_int[0] != '1':
            chinese_num = chinese_num_dict[num_int[0]] + "十" + chinese_num_dict[num_int[1]] + "点" + chinese_num_dict[
                num_int[2]] + chinese_num_dict[num_int[3]] + "度"
        else:
            chinese_num = "十" + chinese_num_dict[num_int[1]] + "点" + chinese_num_dict[num_int[2]] + chinese_num_dict[
                num_int[3]] + "度"
    elif len(num_int) == 3:
        chinese_num = chinese_num_dict[num_int[0]] + "点" + chinese_num_dict[num_int[1]] + chinese_num_dict[
            num_int[2]] + "度"
    else:
        return "温度测量错误"

    return "体温" + chinese_num



if __name__ == '__main__':
    print_hi('PyCharm')
    print(to_zh(36.00))


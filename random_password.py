import random


def tao_password(do_dai: int) -> str:
    password = ""

    for _ in range(do_dai):
        # 1: chữ hoa, 2: chữ thường, 3: chữ số, 4: ký tự đặc biệt
        loai = random.randint(1, 4)

        if loai == 1:  # chữ hoa A-Z
            ma = random.randint(65, 90)
        elif loai == 2:  # chữ thường a-z
            ma = random.randint(97, 122)
        elif loai == 3:  # chữ số 0-9
            ma = random.randint(48, 57)
        else:  # ký tự đặc biệt (từ '!' đến '/')
            ma = random.randint(33, 47)

        ky_tu = chr(ma)
        password += ky_tu

    return password


if __name__ == "__main__":
    do_dai = int(input("Nhập độ dài password: "))
    pw = tao_password(do_dai)
    print("Password ngẫu nhiên:", pw)

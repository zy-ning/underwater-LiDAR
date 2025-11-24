import datetime
import time

import serial

# --- 配置区域 ---
COM_PORT = "COM3"  # 你的端口号
BAUD_RATE = 1000000  # 你的波特率
# ----------------
DIST = 150 - 65  # cm


def run_lidar_test():
    try:
        # 打开串口
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        print(f"✅ 成功打开串口 {COM_PORT}，等待数据中...")

        # --- 发送配置命令 ---
        # 0x55 0x11 0x00 0x01 [Mode] [CRC]
        # Mode: 0x02 (RAW)
        mode = 0x02
        cmd = [0x55, 0x11, 0x00, 0x01, mode]
        crc = sum(cmd) & 0xFF
        cmd.append(crc)

        ser.write(bytes(cmd))
        print(f"📤 发送指令: {' '.join([f'{x:02X}' for x in cmd])}")
        time.sleep(0.1)

        # --- 数据保存状态 ---
        last_save_time = time.time()
        raw_data_buffer = []
        dist_samples = []

        while True:
            # 1. 读取头部
            if ser.read(1) == b"\xaa":
                byte2 = ser.read(1)

                # --- 情况A: 距离数据 (AA 0F) ---
                if byte2 == b"\x0f":
                    data_body = ser.read(8)
                    if len(data_body) == 8:
                        dist_low = data_body[2]
                        dist_high = data_body[3]
                        distance = dist_low | (dist_high << 8)
                        dist_samples.append(distance)

                # --- 情况B: 原始直方图数据 (AA F1) ---
                elif byte2 == b"\xf1":
                    # 读取长度 (2 bytes, MSB LSB)
                    len_bytes = ser.read(2)
                    if len(len_bytes) == 2:
                        payload_len = (len_bytes[0] << 8) | len_bytes[1]

                        # 读取数据体 (Payload) + CRC (1 byte)
                        payload = ser.read(payload_len)
                        crc_byte = ser.read(1)

                        if len(payload) == payload_len:
                            # Payload = Index(1) + StartPos(4) + RawData(...)
                            idx = payload[0]
                            raw_vals = payload[5:]

                            raw_hex = " ".join([f"{b:02X}" for b in raw_vals])
                            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[
                                :-3
                            ]

                            log_line = f"[{timestamp}] IDX:{idx:02X} LEN:{len(raw_vals)} DATA:{raw_hex}"
                            raw_data_buffer.append(log_line)
                            print(f"📥 Raw Packet: Idx={idx} Len={len(raw_vals)}")

            # --- 定时保存 (每2秒) ---
            if time.time() - last_save_time >= 2.0:
                if raw_data_buffer:
                    avg_dist = DIST
                    if dist_samples:
                        avg_dist = sum(dist_samples) / len(dist_samples)
                        dist_samples = []

                    ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"raw_data_{ts_str}_dist{int(avg_dist)}.log"

                    try:
                        with open(filename, "w") as f:
                            f.write("\n".join(raw_data_buffer))
                        print(f"💾 已保存 {len(raw_data_buffer)} 条数据到 {filename}")
                    except Exception as e:
                        print(f"❌ 保存失败: {e}")

                    raw_data_buffer = []

                last_save_time = time.time()

    except serial.SerialException:
        print(f"❌ 无法打开串口 {COM_PORT}，请检查：")
        print("1. SSCOM 串口助手是不是没关？(必须关闭)")
        print("2. USB线是不是拔了？")
    except KeyboardInterrupt:
        print("\n⏹️ 程序已停止")
    finally:
        if "ser" in locals() and ser.is_open:
            ser.close()


if __name__ == "__main__":
    run_lidar_test()

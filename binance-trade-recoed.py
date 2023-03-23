from binance.spot import Spot
import sqlite3
import configparser

# 从配置文件读取api_key和api_secret
config = configparser.ConfigParser()
config.read('config.ini')
api_key = config['keys']['api_key']
api_secret = config['keys']['api_secret']


client = Spot(api_key=api_key, api_secret=api_secret)


def get_klines(symbol, interval, start_time, end_time):
    klines = client.klines(
        symbol, interval, startTime=start_time, endTime=end_time)
    return klines

# 获取一天的开始时间和结束时间timestamp格式为毫秒
def get_day_timestamp(day):
    """
    :param day: 2017-9-1
    :return: 1504253880000, 1610230800000
    """
    day = day.split("-")
    year = int(day[0])
    month = int(day[1])
    day = int(day[2])
    import time
    start_time = time.mktime((year, month, day, 0, 0, 0, 0, 0, 0)) * 1000
    end_time = time.mktime((year, month, day, 23, 59, 59, 0, 0, 0)) * 1000
    return start_time, end_time

# 创建数据表，数据表表名为klines，
# 字段为open_time, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume，ignore
def create_table():
    conn = sqlite3.connect('binance.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE ethusdt
                 (open_time, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume,ignore)''')
    conn.commit()
    conn.close()

# 获取指定时间的k线数据
def get_klines_from_time(symbol, interval, start_time, end_time):
    klines = client.klines(
        symbol, interval, startTime=start_time, endTime=end_time)

    # 保存数据到数据库
    c = conn.cursor()
    for kline in klines:
        c.execute("INSERT INTO ethusdt VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", kline)
    conn.commit()

    if klines[-1][0] < end_time:
        # 继续获取下一时刻数据
        start_time = klines[-1][0] + 60000
        print(f"当前获取数据时间为：{start_time}", sep="\r")
        klines = get_klines_from_time(symbol, interval, start_time, end_time)
    else:
        return


if __name__ == '__main__':
    create_table()
    # 2017-9-1
    conn = sqlite3.connect('binance.db')
    start_time = 1504253880000
    end_time = 1679500740000  # 2023-3-22 23:59:00
    symbol = "ETHUSDT"  # 交易对
    interval = "1m"  # 时间间隔
    get_klines_from_time(symbol, interval, start_time, end_time)
    conn.close()


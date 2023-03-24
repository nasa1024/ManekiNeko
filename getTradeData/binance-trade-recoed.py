from binance.spot import Spot
import sqlite3
import configparser

# 从配置文件读取api_key和api_secret
config = configparser.ConfigParser()
config.read('config.ini')
api_key = config['keys']['api_key']
api_secret = config['keys']['api_secret']


client = Spot(api_key=api_key, api_secret=api_secret)


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
def create_table(symbol="ETHUSDT"):
    conn = sqlite3.connect('binance.db')
    c = conn.cursor()
    c.execute(f'''CREATE TABLE {symbol}
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
        c.execute(f"INSERT INTO {symbol} VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", kline)
    conn.commit()

    while klines[-1][0] < end_time:
        # 继续获取下一时刻数据
        start_time = klines[-1][0] + 60000 * 5
        print(f"当前获取数据时间为：{start_time}", sep="\r")
        klines = client.klines(
            symbol, interval, startTime=start_time, endTime=end_time)
        c = conn.cursor()
        for kline in klines:
            c.execute(
                f"INSERT INTO {symbol} VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", kline)
        conn.commit()
    print("crawl successful")


if __name__ == '__main__':
    # 交易对列表
    symbol_list = ["ETHUSDT", "BTCUSDT", "LTCUSDT", "BCHUSDT", "EOSUSDT", "XRPUSDT", "ETCUSDT", "ADAUSDT",  "ZECUSDT", "BNBUSDT",
                   "LINKUSDT",  "USDCUSDT", "MATICUSDT", "FILUSDT","AXSUSDT","DODOUSDT",
                   "BUSDUSDT",
                   "DOGEUSDT",
                   "SUSHIUSDT",
                   "AVAXUSDT", "DOTUSDT",  "SOLUSDT",  "AAVEUSDT"]
    conn = sqlite3.connect('binance.db')
    for symbol in symbol_list:
        create_table(symbol)
        # 2017-9-1
        start_time = 1504253880000
        end_time = 1679500740000  # 2023-3-22 23:59:00 1679363700000
        symbol = "ETHUSDT"  # 交易对
        interval = "5m"  # 时间间隔
        try:
            get_klines_from_time(symbol, interval, start_time, end_time)
        except Exception as e:
            print(e)
            pass

    conn.close()

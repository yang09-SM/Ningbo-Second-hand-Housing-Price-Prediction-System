import sqlite3
import pandas as pd
import os

def create_connection(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    return conn

def create_table(conn):
    create_table_sql = '''
    CREATE TABLE IF NOT EXISTS houses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        总价 TEXT,
        单价 TEXT,
        小区名称 TEXT,
        所在区域 TEXT,
        房屋户型 TEXT,
        所在楼层 TEXT,
        建筑面积 TEXT,
        户型结构 TEXT,
        套内面积 TEXT,
        建筑类型 TEXT,
        房屋朝向 TEXT,
        建筑结构 TEXT,
        装修情况 TEXT,
        梯户比例 TEXT,
        配备电梯 TEXT,
        挂牌时间 TEXT,
        交易权属 TEXT,
        上次交易 TEXT,
        房屋用途 TEXT,
        房屋年限 TEXT,
        产权所属 TEXT,
        抵押信息 TEXT,
        房本备件 TEXT,
        房源核验码 TEXT,
        用水类型 TEXT,
        用电类型 TEXT,
        燃气价格 TEXT,
        别墅类型 TEXT,
        区 TEXT,
        板块 TEXT,
        楼层位置 TEXT,
        总楼层数 INTEGER,
        室数 INTEGER,
        厅数 INTEGER,
        厨数 INTEGER,
        卫数 INTEGER
    )
    '''
    cursor = conn.cursor()
    cursor.execute(create_table_sql)

    create_users_sql = '''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'user'
    )
    '''
    cursor.execute(create_users_sql)
    conn.commit()

def init_users(conn):
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO users (username, password, role) VALUES ('admin', 'admin123', 'admin')"
    )
    conn.commit()

def import_data(conn, data_path):
    df = pd.read_csv(data_path)
    # 先删除表（如果存在），然后重新创建带id的表，再导入数据
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS houses')
    conn.commit()
    
    # 重新创建表
    create_table(conn)
    
    # 导入数据
    df.to_sql('houses', conn, if_exists='append', index=False)
    conn.commit()
    return len(df)

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    data_path = os.path.join(base_dir, 'data', 'house_data_for_db.csv')
    db_path = os.path.join(base_dir, 'data', 'houses.db')
    
    print("=" * 80)
    print("宁波二手房价格预测 - 数据库初始化")
    print("=" * 80)
    
    print("\n1. 正在创建数据库连接...")
    conn = create_connection(db_path)
    
    print("\n2. 正在创建表结构...")
    create_table(conn)

    print("\n3. 正在初始化用户表...")
    init_users(conn)

    print("\n4. 正在导入数据...")
    count = import_data(conn, data_path)
    print(f"   已导入 {count} 条数据")
    
    conn.close()
    
    print(f"\n" + "=" * 80)
    print(f"数据库初始化完成!")
    print(f"  - 数据库路径: {os.path.abspath(db_path)}")
    print("=" * 80)

if __name__ == "__main__":
    main()

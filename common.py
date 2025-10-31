from datetime import datetime
import pytz

def get_current_japanese_timestamp():
    jst = pytz.timezone("Asia/Tokyo")
    now_jst = datetime.now(jst) 
    timestamp = now_jst.strftime("%Y_%m%d_%H%M")
    
    print(f"日時：{timestamp}")
    return timestamp

if __name__ == "__main__":
    current_japanese_timestamp()
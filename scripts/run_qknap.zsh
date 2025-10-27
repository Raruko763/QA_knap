#!/bin/zsh

count=0
max=10

while (( count < max )); do
  echo "実行 $((count + 1)) 回目..."
  python3 src/Qknapcore.py -j before_data.json -sp . 

  ret=$?  
  if (( ret == 0 )); then
    (( count++ ))
  else
    echo "エラーが発生しました。再試行中..."
  fi
done

echo "完了！"

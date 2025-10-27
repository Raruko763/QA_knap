#!/usr/bin/env ruby
require 'fileutils'
require 'dotenv'
Dotenv.load

# ==========================================================
# 基本設定
# ==========================================================
$save_path = ENV['SAVE_PATH'] || "./out"
FileUtils.mkdir_p($save_path)

# ==========================================================
# 対象インスタンス一覧
# ==========================================================
$instances = [
  "X-n856-k95.vrp",
  "Leuven2.vrp",
  "E-n51-k5.vrp",
  "E-n101-k14.vrp",
  "X-n1001-k43.vrp"
]

# ==========================================================
# 実行ループ
# ==========================================================
$instances.each do |instance|
  # 出力ファイル名（インスタンス名_before_data.json）
  json_name = "#{File.basename(instance, File.extname(instance))}_before_data.json"
  save_path = File.join($save_path, json_name)

  puts "\n========== 実行開始: #{instance} =========="
  cmd = "julia src/QAknapsack.jl #{$save_path} #{instance}"
  puts "[CMD] #{cmd}"

  success = system(cmd)
  if success && File.exist?(save_path)
    puts "✅ 完了: #{instance} → 出力: #{save_path}"
  else
    warn "⚠️ 失敗または出力なし: #{instance}"
  end
end

puts "\n✅ すべてのインスタンスで QAknapsack.jl を実行しました。"

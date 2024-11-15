# 中科院採購案測試資料集生成工具

使用Python開發，

## 資料集格式

假設有$n$種飛彈和$m$種戰艦，則一份資料集包含以下檔案：

* turrets.csv: 岸防砲資訊，每列包含`id`、`x`、`y`、`missle_1_count`、`missle_2_count`、$\cdots$、`missle_n_count`
* ships.csv: 敵艦位置資訊，每列包含`id`、`x`、`y`、`type`
* ship_types.csv: 敵艦種類，每列包含`type`、`require_missle_1_count`、`require_missle_2_count`、$\cdots$、`require_missle_n_count`
  * 不使用飛彈造成傷害量與血量，而是採用這樣的設計，是為了與中科院文件中的描述相符。
* missles.csv: 飛彈資訊，每列包含`type`、`min_range`、`max_range`

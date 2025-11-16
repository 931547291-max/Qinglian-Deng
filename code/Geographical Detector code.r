# 安装并加载必要的 R 包
if (!require("GD")) install.packages("GD")
if (!require("readxl")) install.packages("readxl")
library(GD)
library(readxl)

### 1. 读取数据
cat("请选择包含碳数据及驱动因子的数据文件...\n")
data_path <- file.choose()  

# 根据文件格式读取数据
if (tools::file_ext(data_path) %in% c("xlsx", "xls")) {
  data.GPP <- read_excel(data_path)
} else {
  data.GPP <- read.csv(data_path, header = TRUE, stringsAsFactors = FALSE)
}

### 2. 数据预处理
# 2.1 清理列名
colnames(data.GPP) <- gsub("[^a-zA-Z0-9_]", "_", colnames(data.GPP))  
cat("处理后的列名：", paste(colnames(data.GPP), collapse = ", "), "\n\n")

# 2.2 显示列名供用户选择因变量（碳数据）
cat("请从以下列中选择因变量（碳数据）的编号：\n")
for (i in seq_along(colnames(data.GPP))) {
  cat(sprintf("%d. %s\n", i, colnames(data.GPP)[i]))
}

# 让用户选择因变量
dep_index <- as.integer(readline("请输入因变量（碳数据）的编号: "))
dependent_var <- colnames(data.GPP)[dep_index]
cat("已选择因变量：", dependent_var, "\n\n")

# 2.3 确定驱动因子（排除因变量的其他列）
driver_vars <- setdiff(colnames(data.GPP), dependent_var)  

# 2.4 转换因变量类型为数值型
data.GPP[[dependent_var]] <- as.numeric(data.GPP[[dependent_var]])  
if (any(is.na(data.GPP[[dependent_var]]))) {
  warning("因变量中存在无法转换为数值的记录，后续会被na.omit处理")
}

# 2.5 转换驱动因子类型
for (var in driver_vars) {
  # 尝试转换为数值型，若失败则转为因子型
  num_test <- as.numeric(as.character(data.GPP[[var]]))
  if (sum(is.na(num_test)) / length(num_test) < 0.2) {  # 若大部分能转为数值
    data.GPP[[var]] <- num_test
    cat(var, "已转换为数值型\n")
  } else {
    data.GPP[[var]] <- as.factor(data.GPP[[var]])
    cat(var, "已转换为因子型，水平数：", length(levels(data.GPP[[var]])), "\n")
  }
}

# 2.6 清理缺失值
nrow_before <- nrow(data.GPP)
data.GPP <- na.omit(data.GPP)
cat("\n清理缺失值：原始", nrow_before, "行，清理后保留", nrow(data.GPP), "行\n\n")

### 3. 筛选有效驱动因子
valid_drivers <- sapply(driver_vars, function(var) {
  return(var %in% colnames(data.GPP) && 
           (is.factor(data.GPP[[var]]) || is.numeric(data.GPP[[var]])) && 
           sum(!is.na(data.GPP[[var]])) > 0)  
})
valid_drivers <- names(valid_drivers[valid_drivers])
cat("有效驱动因子：", paste(valid_drivers, collapse = ", "), "\n\n")

### 4. 构建模型公式
formula_str <- paste(dependent_var, "~", paste(valid_drivers, collapse = " + "))
gdm_formula <- as.formula(formula_str)
cat("模型公式：", formula_str, "\n\n")

### 5. 运行地理探测器分析
gppgdm <- gdm(
  formula = gdm_formula,
  continuous_variable = valid_drivers,
  data = data.GPP,
  discmethod = c("equal", "natural", "quantile"),
  discitv = 3:7
)

### 6. 输出结果
cat("===== 地理探测器分析结果 =====\n")
print(gppgdm)

# 可视化结果
cat("\n正在绘制分析结果图...\n")
plot(gppgdm)
title(paste("碳数据(", dependent_var, ")驱动因子分析结果", sep = ""))

# 保存结果
save(gppgdm, file = "Carbon_Geodetector_Results.RData")
cat("\n分析结果已保存为 Carbon_Geodetector_Results.RData\n")
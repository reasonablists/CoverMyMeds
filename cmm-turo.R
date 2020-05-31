#### CoverMyMeds Challenge Problem - ARIMA-GARCH approach

### Load, format, and split the data
# Load data
library(arrow)
coverdata =read_parquet("Documents/Code_Turo_versions/cmm.pq", as_tibble = TRUE)

# Reformat volume data
cmm = coverdata
cmm$volume_A = as.integer(cmm$volume_A)
cmm$volume_B = as.integer(cmm$volume_B)
cmm$volume_C = as.integer(cmm$volume_C)


# Make train and test data splits:
library(lubridate)

# One month
mon1_trdate1 <- as.POSIXct("2016-12-31")
mon1_trdate2 <- as.POSIXct("2019-11-30")
mon1_tr_int <- interval(mon1_trdate1, mon1_trdate2)
mon1_train <- cmm[cmm$date_val %within% mon1_tr_int,]

mon1_test_date1 <- as.POSIXct("2019-11-30")
mon1_test_date2 <- as.POSIXct("2020-01-01")
mon1_test_int <- interval(mon1_test_date1, mon1_test_date2)
mon1_test <- cmm[cmm$date_val %within% mon1_test_int,]  

# Three month
mon3_trdate1 <- as.POSIXct("2016-12-31")
mon3_trdate2 <- as.POSIXct("2019-9-30")
mon3_tr_int <- interval(mon3_trdate1, mon3_trdate2)
mon3_train <- cmm[cmm$date_val %within% mon3_tr_int,]

mon3_test_date1 <- as.POSIXct("2019-09-30")
mon3_test_date2 <- as.POSIXct("2020-01-01")
mon3_test_int <- interval(mon3_test_date1, mon3_test_date2)
mon3_test <- cmm[cmm$date_val %within% mon3_test_int,]  

# Six month
mon6_trdate1 <- as.POSIXct("2016-12-31")
mon6_trdate2 <- as.POSIXct("2019-6-30")
mon6_tr_int <- interval(mon6_trdate1, mon6_trdate2)
mon6_train <- cmm[cmm$date_val %within% mon6_tr_int,]

mon6_test_date1 <- as.POSIXct("2019-07-01")
mon6_test_date2 <- as.POSIXct("2020-01-01")
mon6_test_int <- interval(mon6_test_date1, mon6_test_date2)
mon6_test <- cmm[cmm$date_val %within% mon6_test_int,]  

### 1-month modeling first:

## Examine 1-month training data with:
# 1) ACF (auto-correlation function)
par(mfrow=c(3,1))
acf_a1 = acf(mon1_train$volume_A)
acf_b1 = acf(mon1_train$volume_B)
acf_c1 = acf(mon1_train$volume_C)

# 2) PACF (partial autocorrelations)
par(mfrow=c(3,1))
pacf_a1 = pacf(mon1_train$volume_A)
pacf_b1 = pacf(mon1_train$volume_B)
pacf_c1 = pacf(mon1_train$volume_C)


## Fit ARIMA-GARCH model for 1-month data
library(rugarch)
spec1 <- ugarchspec( 
  variance.model = list(model = "sGARCH", garchOrder = c(1, 3)),
  mean.model = list(armaOrder = c(5, 4), include.mean = TRUE),
  distribution.model = "std")

fit_1a <- ugarchfit(data = mon1_train$volume_A, spec=spec1)
fit_1b <- ugarchfit(data = mon1_train$volume_B, spec=spec1)
fit_1c <- ugarchfit(data = mon1_train$volume_C, spec=spec1)

# Forecast 1 month (31 days)
forecast_1a <- ugarchforecast(fit_1a,n.ahead=31)
forecast_1b <- ugarchforecast(fit_1b,n.ahead=31)
forecast_1c <- ugarchforecast(fit_1c,n.ahead=31)
par(mfrow=c(3,1))
plot(forecast_1a, which = 1)
plot(forecast_1b, which = 1)
plot(forecast_1c, which = 1)

## Fitted and test data to new df and compare
library(tibble)
fitted_mon1a <- forecast_1a@forecast$seriesFor
comb_month1a <- subset(mon1_test, select = c(date_val,volume_A))
comb_month1a <- comb_month1a %>% add_column(as.integer(fitted_mon1a))

fitted_mon1b <- forecast_1b@forecast$seriesFor
comb_month1b <- subset(mon1_test, select = c(date_val,volume_B))
comb_month1b <- comb_month1b %>% add_column(as.integer(fitted_mon1b))

fitted_mon1c <- forecast_1c@forecast$seriesFor
comb_month1c <- subset(mon1_test, select = c(date_val,volume_C))
comb_month1c <- comb_month1c %>% add_column(as.integer(fitted_mon1c))

# Calculate fit with MAPE
library(MLmetrics)
mape1a <- MAPE(y_pred = comb_month1a$`as.integer(fitted_mon1a)`,
               y_true = comb_month1a$volume_A)
mape1a
mape1b <- MAPE(y_pred = comb_month1b$`as.integer(fitted_mon1b)`,
               y_true = comb_month1b$volume_B)
mape1b 
mape1c <- MAPE(y_pred = comb_month1c$`as.integer(fitted_mon1c)`,
               y_true = comb_month1c$volume_C)
mape1c

# Plot 1 month forecasts
library(ggplot2)
library(gridExtra)
library(RColorBrewer)

gplot_1a <- ggplot() +
  geom_line(data = comb_month1a, aes(x=date_val, y=volume_A, color ="Test A"),size = .5) +
  geom_line(data = comb_month1a, aes(x=date_val, y=fitted_mon1a, color = "Fitted A"),size = .5) +
  geom_line(data = mon1_train, aes(x=date_val, y=volume_A, color = "Train A"),size = .5) +
  scale_x_date(limits = as.Date(c("2019-09-01","2019-12-31")),
               date_labels = "%b %Y",
               date_breaks = "1 month",
               date_minor_breaks = "1 week") +
  scale_color_brewer(palette = "Set2") +
  ggtitle("1-month forecast with ARMA-GARCH(1,3),(5,4)") +
  xlab("") +
  ylab("Volume A") +
  labs(colour = "")

gplot_1b <- ggplot() +
  geom_line(data = comb_month1b, aes(x=date_val, y=volume_B, color ="Test B"), size = .5) +
  geom_line(data = comb_month1b, aes(x=date_val, y=fitted_mon1b, color = "Fitted B"),size = .5) +
  geom_line(data = mon1_train, aes(x=date_val, y=volume_B, color = "Train B"),size = .5) +
  scale_x_date(limits = as.Date(c("2019-09-01","2019-12-31")),
               date_labels = "%b %Y",
               date_breaks = "1 month",
               date_minor_breaks = "1 week") +  xlab("") +
  scale_color_brewer(palette = "Set2") +
  ylab("Volume B") +
  labs(colour = "")

gplot_1c <- ggplot() +
  geom_line(data = comb_month1c, aes(x=date_val, y=volume_C, color ="Test C"),size = .5) +
  geom_line(data = comb_month1c, aes(x=date_val, y=fitted_mon1c, color = "Fitted C"),size = .5) +
  geom_line(data = mon1_train, aes(x=date_val, y=volume_C, color = "Train C"),size = .5) +
  scale_x_date(limits = as.Date(c("2019-09-01","2019-12-31")),
               date_labels = "%b %Y",
               date_breaks = "1 month",
               date_minor_breaks = "1 week") +  xlab("") +
  scale_color_brewer(palette = "Set2") +
  ylab("Volume C") +
  labs(colour = "")

gplot_grid_1 <- grid.arrange(gplot_1a, gplot_1b, gplot_1c, nrow=3)

### 3-month modeling next:

## Examine 3-month training data with:
# 1) ACF (auto-correlation function)
par(mfrow=c(3,1))
acf_a3 = acf(mon3_train$volume_A)
acf_b3 = acf(mon3_train$volume_B)
acf_c3 = acf(mon3_train$volume_C)

# 2) PACF (partial autocorrelations)
par(mfrow=c(3,1))
pacf_a3 = pacf(mon3_train$volume_A)
pacf_b3 = pacf(mon3_train$volume_B)
pacf_c3 = pacf(mon3_train$volume_C)


## Fit ARIMA-GARCH model for 3-month data
library(rugarch)
spec3 <- ugarchspec( 
  variance.model = list(model = "sGARCH", garchOrder = c(1, 5)),
  mean.model = list(armaOrder = c(3, 3), include.mean = TRUE),
  distribution.model = "std")

fit_3a <- ugarchfit(data = mon3_train$volume_A, spec=spec3)
fit_3b <- ugarchfit(data = mon3_train$volume_B, spec=spec3)
fit_3c <- ugarchfit(data = mon3_train$volume_C, spec=spec3)

# Forecast 3 month (92 days)
forecast_3a <- ugarchforecast(fit_3a,n.ahead=92)
forecast_3b <- ugarchforecast(fit_3b,n.ahead=92)
forecast_3c <- ugarchforecast(fit_3c,n.ahead=92)
par(mfrow=c(3,1))
plot(forecast_3a, which = 1)
plot(forecast_3b, which = 1)
plot(forecast_3c, which = 1)

## Fitted and test data to new df and compare
library(tibble)
fitted_mon3a <- forecast_3a@forecast$seriesFor
comb_month3a <- subset(mon3_test, select = c(date_val,volume_A))
comb_month3a <- comb_month3a %>% add_column(as.integer(fitted_mon3a))

fitted_mon3b <- forecast_3b@forecast$seriesFor
comb_month3b <- subset(mon3_test, select = c(date_val,volume_B))
comb_month3b <- comb_month3b %>% add_column(as.integer(fitted_mon3b))

fitted_mon3c <- forecast_3c@forecast$seriesFor
comb_month3c <- subset(mon3_test, select = c(date_val,volume_C))
comb_month3c <- comb_month3c %>% add_column(as.integer(fitted_mon3c))

# Calculate fit with MAPE
mape3a <- MAPE(y_pred = comb_month3a$`as.integer(fitted_mon3a)`,
               y_true = comb_month3a$volume_A)
mape3a
mape3b <- MAPE(y_pred = comb_month3b$`as.integer(fitted_mon3b)`,
               y_true = comb_month3b$volume_B)
mape3b
mape3c <- MAPE(y_pred = comb_month3c$`as.integer(fitted_mon3c)`,
               y_true = comb_month3c$volume_C)
mape3c

# Plot the 3-month forecasts versus the test data
gplot_3a <- ggplot() +
  geom_line(data = comb_month3a, aes(x=date_val, y=volume_A, color ="Test A"),size = .5) +
  geom_line(data = comb_month3a, aes(x=date_val, y=fitted_mon3a, color = "Fitted A"),size = .5) +
  geom_line(data = mon3_train, aes(x=date_val, y=volume_A, color = "Train A"),size = .5) +
  scale_x_date(limits = as.Date(c("2019-07-01","2019-12-31")),
               date_labels = "%b %Y",
               date_breaks = "1 month") +
  scale_color_brewer(palette = "Set2") +
  ggtitle("3-month forecast with ARMA-GARCH(1,5),(3,3)") +
  xlab("") +
  ylab("Volume A") +
  labs(colour = "")

gplot_3b <- ggplot() +
  geom_line(data = comb_month3b, aes(x=date_val, y=volume_B, color ="Test B"), size = .5) +
  geom_line(data = comb_month3b, aes(x=date_val, y=fitted_mon3b, color = "Fitted B"),size = .5) +
  geom_line(data = mon3_train, aes(x=date_val, y=volume_B, color = "Train B"),size = .5) +
  scale_x_date(limits = as.Date(c("2019-07-01","2019-12-31")),
               date_labels = "%b %Y",
               date_breaks = "1 month") +  
  xlab("") +
  scale_color_brewer(palette = "Set2") +
  ylab("Volume B") +
  labs(colour = "")

gplot_3c <- ggplot() +
  geom_line(data = comb_month3c, aes(x=date_val, y=volume_C, color ="Test C"),size = .5) +
  geom_line(data = comb_month3c, aes(x=date_val, y=fitted_mon3c, color = "Fitted C"),size = .5) +
  geom_line(data = mon3_train, aes(x=date_val, y=volume_C, color = "Train C"),size = .5) +
  scale_x_date(limits = as.Date(c("2019-07-01","2019-12-31")),
               date_labels = "%b %Y",
               date_breaks = "1 month") +  
  xlab("") +
  scale_color_brewer(palette = "Set2") +
  ylab("Volume C") +
  labs(colour = "")

gplot_grid_3 <- grid.arrange(gplot_3a, gplot_3b, gplot_3c, nrow=3)

### 6-month modeling last:

## Examine 6-month training data with:
# 1) ACF (auto-correlation function)
par(mfrow=c(3,1))
acf_a6 = acf(mon6_train$volume_A)
acf_b6 = acf(mon6_train$volume_B)
acf_c6 = acf(mon6_train$volume_C)

# 2) PACF (partial autocorrelations)
par(mfrow=c(3,1))
pacf_a6 = pacf(mon6_train$volume_A)
pacf_b6 = pacf(mon6_train$volume_B)
pacf_c6 = pacf(mon6_train$volume_C)


## Fit ARIMA-GARCH model for 6-month data
library(rugarch)
spec6 <- ugarchspec( 
  variance.model = list(model = "sGARCH", garchOrder = c(2, 4)),
  mean.model = list(armaOrder = c(3, 5), include.mean = TRUE),
  distribution.model = "std")

fit_6a <- ugarchfit(data = mon6_train$volume_A, spec=spec6)
fit_6a 
fit_6b <- ugarchfit(data = mon6_train$volume_B, spec=spec6)
fit_6c <- ugarchfit(data = mon6_train$volume_C, spec=spec6)

# Forecast 6 month (183 days)
forecast_6a <- ugarchforecast(fit_6a,n.ahead=183)
forecast_6b <- ugarchforecast(fit_6b,n.ahead=183)
forecast_6c <- ugarchforecast(fit_6c,n.ahead=183)

par(mfrow=c(3,1))
plot(forecast_6a, which = 1)
plot(forecast_6b, which = 1)
plot(forecast_6c, which = 1)

## Fitted and test data to new df and compare
library(tibble)
fitted_mon6a <- forecast_6a@forecast$seriesFor
comb_month6a <- subset(mon6_test, select = c(date_val,volume_A))
comb_month6a <- comb_month6a %>% add_column(as.integer(fitted_mon6a))

fitted_mon6b <- forecast_6b@forecast$seriesFor
comb_month6b <- subset(mon6_test, select = c(date_val,volume_B))
comb_month6b <- comb_month6b %>% add_column(as.integer(fitted_mon6b))

fitted_mon6c <- forecast_6c@forecast$seriesFor
comb_month6c <- subset(mon6_test, select = c(date_val,volume_C))
comb_month6c <- comb_month6c %>% add_column(as.integer(fitted_mon6c))

# Calculate fit with MAPE
library(MLmetrics)
mape6a <- MAPE(y_pred = comb_month6a$`as.integer(fitted_mon6a)`,
               y_true = comb_month6a$volume_A)
mape6a
mape6b <- MAPE(y_pred = comb_month6b$`as.integer(fitted_mon6b)`,
               y_true = comb_month6b$volume_B)
mape6b
mape6c <- MAPE(y_pred = comb_month6c$`as.integer(fitted_mon6c)`,
               y_true = comb_month6c$volume_C)
mape6c

# Plot the 6-month forecasts versus the test data
library(ggplot2)
gplot_6a <- ggplot() +
  geom_line(data = comb_month6a, aes(x=date_val, y=volume_A, color ="Test A"),size = .5) +
  geom_line(data = comb_month6a, aes(x=date_val, y=fitted_mon6a, color = "Fitted A"),size = .5) +
  geom_line(data = mon6_train, aes(x=date_val, y=volume_A, color = "Train A"),size = .5) +
  scale_x_date(limits = as.Date(c("2019-01-01","2019-12-31")),
               date_labels = "%b %Y",
               date_breaks = "1 month") +
  scale_color_brewer(palette = "Set2") +
  ggtitle("6-month forecast with ARMA-GARCH(2,4),(3,5)") +
  xlab("") +
  ylab("Volume A") +
  labs(colour = "")

gplot_6b <- ggplot() +
  geom_line(data = comb_month6b, aes(x=date_val, y=volume_B, color ="Test B"), size = .5) +
  geom_line(data = comb_month6b, aes(x=date_val, y=fitted_mon6b, color = "Fitted B"),size = .5) +
  geom_line(data = mon6_train, aes(x=date_val, y=volume_B, color = "Train B"),size = .5) +
  scale_x_date(limits = as.Date(c("2019-01-01","2019-12-31")),
               date_labels = "%b %Y",
               date_breaks = "1 month") +  
  xlab("") +
  scale_color_brewer(palette = "Set2") +
  ylab("Volume B") +
  labs(colour = "")

gplot_6c <- ggplot() +
  geom_line(data = comb_month6c, aes(x=date_val, y=volume_C, color ="Test C"),size = .5) +
  geom_line(data = comb_month6c, aes(x=date_val, y=fitted_mon6c, color = "Fitted C"),size = .5) +
  geom_line(data = mon6_train, aes(x=date_val, y=volume_C, color = "Train C"),size = .5) +
  scale_x_date(limits = as.Date(c("2019-01-01","2019-12-31")),
               date_labels = "%b %Y",
               date_breaks = "1 month") +  
  xlab("") +
  scale_color_brewer(palette = "Set2") +
  ylab("Volume C") +
  labs(colour = "")

gplot_grid_6 <- grid.arrange(gplot_6a, gplot_6b, gplot_6c, nrow=3)












### Output all MAPE for all models:

df1a <- data_frame("Volume A", "1 Month", mape1a)
df3a <- data_frame("Volume A", "3 Months", mape3a)
df6a <- data_frame("Volume A", "6 Months", mape6a)
df1b <- tibble("Volume B", "1 Month", mape1b)
df3b <- tibble("Volume B", "3 Months", mape3b)
df6b <- tibble("Volume B", "6 Months", mape6b)
df1c <- tibble("Volume C", "1 Month", mape1c)
df3c <- tibble("Volume C", "3 Months", mape3c)
df6c <- tibble("Volume C", "6 Months", mape6c)



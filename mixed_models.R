#library(h5)
library(broom.mixed)
library(ggplot2)
library(dplyr)

data <- data.frame(read.table("~/../data/ukbb/processed/activity_by_day.txt", header=TRUE, sep="\t"))
#ukbb <- h5file("~/../data/ukbb/processed/ukbb_data_table.h5")

selected_IDs <- sample(unique(data$ID), 10000)
data <- data[which(data$ID %in% selected_IDs),]

data$date <- as.Date(data$X, "%Y-%m-%d")
data$day_of_week <- as.factor(weekdays(data$date))
data$day <- as.factor(data$date)

VAR <- sym("main_sleep_onset")

mean_by_day <- data %>% group_by(day) %>% summarise_at(vars(!!VAR), list(~mean(., na.rm=TRUE)))

library(lme4)

#model <- lmer(main_sleep_duration ~ day_of_week + scale(main_sleep_onset,)  + (1|day) + (1 + scale(main_sleep_onset, scale=FALSE) |ID), data=data)
model <- lmer(paste(VAR, " ~ day_of_week + (1|day) + (1|ID)"), data=data)
summary(model)

fit <- augment(model, data=data)

coefs <- coef(model)
coefs$day$day <- as.factor(row.names(coefs$day))
coefs$day$date <- as.Date(row.names(coefs$day))
coefs$day$intercept <- coefs$day[,"(Intercept)"]
by_day <- merge(coefs$day, mean_by_day, by.x="day", by.y="day", all.x=TRUE)

g <- ggplot(data=by_day, aes(x=date)) +
      geom_line(aes(y=intercept)) +
      geom_line(aes_string(y=VAR), color="red")+
      ylim(22, ) +
      xlim(as.Date("2014-03-15"), as.Date("2014-04-15")) 
plot(g)

g2 <- ggplot(data=fit, aes(x=date)) +
          geom_point(aes(y=.fitted)) +
          geom_line(data=by_day, aes_string(x="date", y=VAR))+
          xlim(as.Date("2014-03-15"), as.Date("2014-04-15"))
plot(g2)

g3 <- ggplot(data=by_day) + geom_density(aes(x=intercept))
plot(g3)

setwd("/home/virtualvikings/Work/repositories/pystats_utils_base/test/database")

df = read.csv("example_r.csv", sep = ",")
df = df[df$Diabetes_012 != 0,]

bartlett.test(df$Body_mass_index ~ df$Diabetes_012)$p.value
bartlett.test(df$Health_scale ~ df$Diabetes_012)$p.value
bartlett.test(df$Days_bad_mental_health_pre30d ~ df$Diabetes_012)$p.value
bartlett.test(df$Days_bad_physical_health_pre30d ~ df$Diabetes_012)$p.value
bartlett.test(df$Age_scale ~ df$Diabetes_012)$p.value
bartlett.test(df$Education_level ~ df$Diabetes_012)$p.value
bartlett.test(df$Income_scale ~ df$Diabetes_012)$p.value

# Statistical Analysis of eigentlich-Data in "DF2.csv"

library('Matrix')
library('lme4')
library('stringr')
library('boot')
library('visreg')

setwd("~/Desktop/KEC/Korrigiert")
DF2 = read.csv('DF2.csv', stringsAsFactors = FALSE)

############################## Plots #########################################################


plot(density(log(DF2$Pr_B.variation._given_A1)))
plot(density(log(DF2$Pr_B.variation._given_C1)))
plot(density(DF2$speech_rate))
plot(density(DF2$A1_Entropy))
plot(density(DF2$C1_Entropy))

plot(DF2$l.P_B_gvn_A1, DF2$Count_phonemes_eig)
abline(lm(Count_phonemes_eig ~ l.P_B_gvn_A1, data=DF2), col="red")

plot(DF2$l.P_B_gvn_C1, DF2$Count_phonemes_eig)
abline(lm(Count_phonemes_eig ~ l.P_B_gvn_C1, data=DF2), col="red")

plot(DF2$l.P_B_gvn_A1, DF2$Difference_count)
abline(lm(Difference_count ~ l.P_B_gvn_A1, data=DF2), col="red")

plot(DF2$l.P_B_gvn_C1, DF2$Difference_count)
abline(lm(Difference_count ~ l.P_B_gvn_C1, data=DF2), col="red")

DF2[DF2$Difference_count>7,]

plot(DF2$syllable_count, DF2$Speak_duration_of_B)
abline(lm(Speak_duration_of_B ~ syllable_count, data = DF2), col="red")

plot(DF2$Difference_count, DF2$Speak_duration_of_B)
abline(lm(Speak_duration_of_B ~ Difference_count, data = DF2), col="red")

plot(DF2$RelFreq_C1, DF2$C1_Surprisal)
abline(lm(C1_Surprisal ~ RelFreq_C1, data = DF2), col="red")
plot(DF2$RelFreq_A1, DF2$A1_Surprisal)
abline(lm(A1_Surprisal ~ RelFreq_A1, data = DF2), col="red")

################### Analyse ############################################################################

mean(DF2$Difference_count) # the mean difference count is ca. 3
mean(DF2$Speak_duration_of_B) # the mean duration of eigentlich is ca. 312ms
tapply(DF2$Speak_duration_of_B, DF2$Difference_count, mean) # mean duration of each difference count

sum(str_count(DF2$Word_A1, "<P>")) / length(DF2$Word_A1) # prob of preceding pause (ca. 9%)
sum(str_count(DF2$Word_C1, "<P>")) / length(DF2$Word_C1) # prob of following pause (ca. 15%)

length(DF2$LS_in_Vari.[DF2$LS_in_Vari.==TRUE]) / length(DF2$LS_in_Vari.) # prob feature sharing with last segment of preceding word (ca. 24%)
length(DF2$FS_in_Vari.[DF2$FS_in_Vari.==TRUE]) / length(DF2$FS_in_Vari.) # prob feature sharing with first segment of following word (ca. 22%)

table(DF2$syllable_count) / length(DF2$Variation) # percentage of syllables: 1=30,8%, 2=65,7%, 3=3,4%


plot(DF2$Difference_count, DF2$Speak_duration_of_B)
abline(lm(Speak_duration_of_B ~ Difference_count, data = DF2), col="red") # the higher the difference count, the shorter word duration

table(DF2$Difference_count == 3)

###############################Statistical Analysis (lmer and glmer) ####################################################### 

############### models with difference count as outcome (dependant variable) ####################

mod0 = lmer(Difference_count~
              + speech_rate
            + (1|Speaker), data = DF2)
summary(mod0)

mod1.1 = lmer(Difference_count~
                + speech_rate
              + l.P_B_gvn_C1
              #+ C1_Entropy
              #+ C1_Surprisal 
              #+ l.P_B_gvn_A1
              + A1_Entropy
              #+ A1_Surprisal
              + (1|Speaker), data = DF2)
summary(mod1.1)

mod1.2 = lmer(Difference_count~
                + speech_rate
              + l.P_B_gvn_C1
              + C1_Entropy
              #+ C1_Surprisal 
              #+ l.P_B_gvn_A1
              #+ A1_Entropy 
              + A1_Surprisal
              + Pause_C1 
              #+ Pause_A1
              + LS_in_Vari.
              #+ FS_in_Vari.
              + (1|Speaker), data = DF2)
summary(mod1.2)

mod1.3 = lmer(Difference_count~
                + speech_rate
              + l.P_B_gvn_C1
              + C1_Entropy
              #+ C1_Surprisal 
              #+ l.P_B_gvn_A1
              #+ A1_Entropy
              + A1_Surprisal
              + Pause_C1 
              #+ Pause_A1
              + LS_in_Vari.
              + FS_in_Vari.
              + (1|Speaker), data = DF2)
summary(mod1.3) ## --> THIS MODEL INCLUDED IN PAPER 

AIC(mod1.2, mod1.3)
anova(mod1.2, mod1.3) # --> mod1.3 slightly better with p<0.05

# QQ-Plot 
plot(density(residuals(mod1.3)))
qqnorm(residuals(mod1.3))
qqline(residuals(mod1.3))


#### CHECKING specific LS and FS

mod1.3.1 = lmer(Difference_count~
                  + speech_rate
                + l.P_B_gvn_C1
                + C1_Entropy
                #+ C1_Surprisal 
                #+ l.P_B_gvn_A1
                #+ A1_Entropy 
                + A1_Surprisal
                + Pause_C1 
                #+ Pause_A1
                + LS_voiced
                #+ LS_frontvowel
                + LS_plosive
                #+ LS_nasal
                #+ LS_lateral
                + FS_in_Vari.
                + (1|Speaker), data = DF2)
summary(mod1.3.1) ### --> THis MODEL INCLUDED IN PAPER 

AIC(mod1.3, mod1.3.1)
anova(mod1.3, mod1.3.1) # --> mod1.3.1 slightly better with p<0.01

mod1.3.2 = lmer(Difference_count~
                  + speech_rate
                + l.P_B_gvn_C1
                + C1_Entropy
                #+ C1_Surprisal 
                #+ l.P_B_gvn_A1
                #+ A1_Entropy 
                + A1_Surprisal
                + Pause_C1 
                #+ Pause_A1
                + LS_voiced
                #+ LS_frontvowel
                + LS_plosive
                #+ LS_nasal
                #+ LS_lateral
                + FS_voiced
                + (1|Speaker), data = DF2)
summary(mod1.3.2)



# QQ-Plot 
plot(density(residuals(mod1.3.1)))
qqnorm(residuals(mod1.3.1))
qqline(residuals(mod1.3.1))

############################ Models with Speak duration of variation as outcome ####################

mod00 = lmer(Speak_duration_of_B~
               + speech_rate
             + (1|Speaker), data = DF2)
summary(mod00)


mod001.3 = lmer(Speak_duration_of_B~
                  + speech_rate
                + l.P_B_gvn_C1
                #+ C1_Entropy
                #+ C1_Surprisal 
                #+ l.P_B_gvn_A1
                #+ A1_Entropy 
                #+ A1_Surprisal
                + Pause_C1 
                + Pause_A1
                + LS_in_Vari.
                #+ FS_in_Vari.
                + (1|Speaker), data = DF2)
summary(mod001.3) ## --> THIS MODEL INCLUDED IN PAPER

mod001.3.1 = lmer(Speak_duration_of_B~
                    + speech_rate
                  + l.P_B_gvn_C1
                  #+ C1_Entropy
                  #+ C1_Surprisal 
                  #+ l.P_B_gvn_A1
                  #+ A1_Entropy 
                  + A1_Surprisal
                  + Pause_C1 
                  #+ Pause_A1
                  + LS_in_Vari.
                  #+ FS_in_Vari.
                  + (1|Speaker), data = DF2)
summary(mod001.3.1)

AIC(mod001.3, mod001.3.1)
anova(mod001.3, mod001.3.1)

# QQ-Plot 
plot(density(residuals(mod001.3)))
qqnorm(residuals(mod001.3))
qqline(residuals(mod001.3))

#### CHECKING specific LS and FS

mod001.3.2 = lmer(Speak_duration_of_B~
                    + speech_rate
                  + l.P_B_gvn_C1
                  #+ C1_Entropy
                  #+ C1_Surprisal 
                  #+ l.P_B_gvn_A1
                  #+ A1_Entropy 
                  #+ A1_Surprisal
                  + Pause_C1 
                  + Pause_A1
                  + LS_voiced
                  #+ LS_frontvowel
                  #+ LS_nasal
                  #+ LS_lateral
                  #+ FS_in_Vari.
                  + (1|Speaker), data = DF2)
summary(mod001.3.2)

AIC(mod001.3, mod001.3.2)
anova(mod001.3, mod001.3.2) # mod001.3 slightly better 

mod001.3.3 = lmer(Speak_duration_of_B~
                    + speech_rate
                  + l.P_B_gvn_C1
                  #+ C1_Entropy
                  #+ C1_Surprisal 
                  #+ l.P_B_gvn_A1
                  #+ A1_Entropy 
                  #+ A1_Surprisal
                  + Pause_C1 
                  + Pause_A1
                  + LS_voiced
                  #+ LS_frontvowel
                  #+ LS_nasal
                  #+ LS_lateral
                  #+ FS_fricative
                  #+ FS_nasal
                  + FS_plosive 
                  + (1|Speaker), data = DF2)
summary(mod001.3.3)

AIC(mod001.3.2, mod001.3.3)
AIC(mod001.3, mod001.3.3)
anova(mod001.3, mod001.3.3) # mod001.3 slightly better 



#################### Modell für binominale Analysen ###################

####### FS_in_Vari. as dependant variable 
modbin0 = glmer(FS_in_Vari.~
                  + speech_rate
                + (1|Speaker), data = DF2, family='binomial')
summary(modbin0)

inv.logit(-2.12)
visreg(modbin0, 'speech_rate', trans = inv.logit)

modbin = glmer(FS_in_Vari.~
                 + speech_rate
               #+ l.P_B_gvn_A1
               #+ l.P_B_gvn_C1
               #+ A1_Entropy
               + C1_Surprisal
               #+ A1_Surprisal
               #+ C1_Entropy
               #+ LS_in_Vari.
               #+ Pause_A1
               #+ Pause_C1 
               #+ LS_voiced
               + (1|Speaker), data = DF2, family='binomial')
summary(modbin) ## --> THIS MODEL INCLUDED IN PAPER 

AIC(modbin0, modbin)
anova(modbin0, modbin)


######### Model with LS_in_Vari as outcome 

modbin1 = glmer(LS_in_Vari.~
                  + speech_rate # not significant !!!!!
                + (1|Speaker), data = DF2, family='binomial')
summary(modbin1)


modbin1.1 = glmer(LS_in_Vari.~
                    #+ speech_rate
                  + l.P_B_gvn_A1
                  + l.P_B_gvn_C1
                  + (1|Speaker), data = DF2, family='binomial')
summary(modbin1.1)


modbin1.2 = glmer(LS_in_Vari.~
                    + speech_rate
                  + l.P_B_gvn_C1
                  + A1_Entropy
                  + A1_Surprisal
                  + l.P_B_gvn_A1
                  #+ C1_Entropy
                  #+ Pause_A1
                  #+ Pause_C1
                  #+ FS_in_Vari.
                  #+ Speak_duration_of_B 
                  + (1|Speaker), data = DF2, family='binomial')
summary(modbin1.2) ## --> THIS MODEL INCLUDED IN PAPER 


AIC(modbin1.1, modbin1.2)
anova(modbin1.1, modbin1.2) # modbin1.2 better fit with p<0.001












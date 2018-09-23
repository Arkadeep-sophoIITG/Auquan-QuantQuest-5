##################################################################################
##################################################################################
## Template file for problem 1. 						##
## First, fill in your answer logic below					##
##################################################################################
#                                LOGIC GOES BELOW                     		#
##################################################################################
#For expected sum of the roll values:
# If one rolls a 6, the expected sum would be 6 because no matter what is 
# the next roll, it will terminate. If one rolls a 5, then the expected sum would 
# be 5 + (1/6)*6 because one has 1/6 chance of rolling a 6. So, similarly,
# expected sum (in a n sided die) = highest roll value till now + (1/n)*(the roll values greater than the highest roll value). 
# It comes out to be n.
#
# For expected number of rolls:
# It can be noted that nCk out of the n^k possible results of the k rolls are strictly increasing sequences. 
# The expected value of K of rolls excluding the last roll which is a non-increasing roll,
# P(K>k) = (nCk)*(1/n)^k (summed from 0 to n) - 1 = (1+1/n)^n -1 , which goes to e-1 as n->infinity.
##################################################################################
##################################################################################
## You have to fill in two functions BELOW 					##
## Both take in input n = number of sides in the die 				##
## 										##									##
## 1. findSumDieRoll(n)	: Return expected value of sum 				##
## 2. findNumberOfRolls(n)  : Return expected value of number of rolls 		##
## 										##									##
## For both, you only have to fill in the math function where indicated     	##
## 										##									##
## You can run this template file to see the output of your functions       	##
## for a 6 sided die.								##
## Simply run: `python problem1_template.py`                            	##
## You should see the output printed once your program runs.                	##
##                                                                          	##
## DO NOT CHANGE ANYTHING ELSE BELOW. ONLY FILL IN THE LOGIC.      		##
##                                                                          	##
## Good Luck!                                                               	##
##################################################################################
##################################################################################


def findSumDieRoll(n):
	##################################
	##          FILL ME IN          ##
	##################################

	sumRolls = n # Replace me with your answer

	return round(sumRolls, 2)

def findNumberOfRolls(n):
	##################################
	##          FILL ME IN          ##
	##################################

	numRolls = (1+(1/n))**n -1	# Replace me with your answer

	return round(numRolls, 2)

if __name__ == "__main__":	
	numberOfSides = 6.0
	sumOfRolls = findSumDieRoll(numberOfSides)
	numberOfRolls = findNumberOfRolls(numberOfSides)
	print('For a %i-sided die, expected value of sum: %.2f and number of rolls: %.2f'%(numberOfSides, sumOfRolls, numberOfRolls))

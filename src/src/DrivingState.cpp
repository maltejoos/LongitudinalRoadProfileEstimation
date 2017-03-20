/*
 * DrivingState.cpp
 *
 *  Created on: 11.08.2011
 *      Author: joos
 */

#include "DrivingState.h"

DrivingState::DrivingState()
{
	state = straight;
	steeringAngle = 0;
}

void DrivingState::update(int sequence, int imgIndex)
{
	if(sequence==1)
	{
		if((imgIndex > 385 && imgIndex < 435) || (imgIndex > 900 && imgIndex < 990)){
			state = rightTurn;
			steeringAngle = 45;
		}
		else if((imgIndex > 55 && imgIndex < 125)){
			state = leftTurn;
			steeringAngle = -45;
		}
		else{
			state = straight;
			steeringAngle = 0;
		}
	}

	else if(sequence==2)
	{
		if((imgIndex > 490 && imgIndex < 540)){
			state = rightTurn;
			steeringAngle = 45;
		}
		else if((imgIndex > 120 && imgIndex < 200)){
			state = leftTurn;
			steeringAngle = -45;
		}
		else{
			state = straight;
			steeringAngle = 0;
		}

	}

	else if(sequence==3)
	{
		if((imgIndex > 30 && imgIndex < 60) || (imgIndex > 340)){
			state = rightTurn;
			steeringAngle = 45;
		}
		else if((imgIndex > 135 && imgIndex < 170) || (imgIndex > 275 && imgIndex < 310)){
			state = leftTurn;
			steeringAngle = -45;
		}
		else{
			state = straight;
			steeringAngle = 0;
		}

	}

	else if(sequence==4)
	{
		if((imgIndex > 285 && imgIndex < 350) || (imgIndex > 190 && imgIndex < 230)){
			state = rightTurn;
			steeringAngle = 45;
		}
		else{
			state = straight;
			steeringAngle = 0;
		}
	}
}

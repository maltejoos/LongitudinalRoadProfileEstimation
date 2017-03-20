/*
 * DrivingState.h
 *
 *  Created on: 11.08.2011
 *      Author: joos
 */

#ifndef DRIVINGSTATE_H_
#define DRIVINGSTATE_H_

struct DrivingState
{
	enum STATE {leftTurn, rightTurn, straight};

	STATE state;
	double steeringAngle;

	DrivingState();

	void update(int sequence, int imgIndex);
};

#endif /* DRIVINGSTATE_H_ */

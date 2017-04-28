package jREC;

import java.awt.Graphics;
import java.awt.Color;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class CartPole extends Environment{

	// Observations:
	// o[0]: angle diff in rads to food
	// o[1]: normalized distance to food

	// Params
	private final double gravity = 9.8;
	private final double massCart = 1.0;
	private final double massPole = 0.1;
	private final double totalMass = massPole + massCart;
	private final double length = 0.5;
	private final double polemassLength = massPole * length;
	private final double forceMag = 10.0;
	private final double tau = 0.02;
	private final double thetaThresholdRadians = 12 * 2 * Math.PI / 360;
	private final double xThreshold = 2.4;

	double[] highLimit = {xThreshold * 2, 99999, thetaThresholdRadians, 99999};


	// Actions:
	// 0: change dir left
	// 1: change dir right
	// 2: move forward
	public static class Action extends Info{
		public int action;
	}

	public static class CartPoleObservation extends Info{
		double[] data;

		@Override
		public double[] getDouble1(){
			return data;
		}
	}

	static class CartPoleState extends Info{
		double x, xDot, theta, thetaDot;
	}

	private CartPoleObservation getObservation(CartPoleState state){
		CartPoleObservation ro = new CartPoleObservation();
		ro.data = new double[4];
		ro.data[0] = state.x;
		ro.data[1] = state.xDot;
		ro.data[2] = state.theta;
		ro.data[3] = state.thetaDot;
		return ro;
	}

	// Returns amount/type of info available
	public Space getObservationSpace(){
		Space space = new Space();
		space.type = Space.Type.CONTINUOUS;
		space.dimensions = new int[]{4};
		return space;
	}

	// Returns number of actions available
	public Space getActionSpace(){
		Space space = new Space();
		space.type = Space.Type.DISCRETE;
		space.dimensions = new int[]{2};
		return space;
	}

	// Resets the environment, returns initial observation
	public Info reset(){
		CartPoleState rs = new CartPoleState();
		state = rs;
		rs.x = (Math.random() * .1) - .05;
		rs.xDot = (Math.random() * .1) - .05;
		rs.theta = (Math.random() * .1) - .05;
		rs.thetaDot = (Math.random() * .1) - .05;
		return getObservation((CartPoleState)state);
	}

	// Implementation for next state
	protected ROF nextState(Info infoState, Info infoAction){
		ROF rof = new ROF();
		CartPoleState state = (CartPoleState)infoState;
		rof.nextState = state;

		int action = ((Action)infoAction).action;

		if(action != 0 && action != 1){
			System.out.println("Problem");
		}

        double force = action == 1 ? forceMag : -forceMag;
        double cosTheta = Math.cos(state.theta);
        double sinTheta = Math.sin(state.theta);
        double temp = (force + polemassLength * state.thetaDot * state.thetaDot * sinTheta) / totalMass;
        double thetaAcc = (gravity * sinTheta - cosTheta * temp) / (length * (4.0/3.0 - massPole * Math.pow(cosTheta, 2) / totalMass));
        double xAcc  = temp - polemassLength * thetaAcc * cosTheta / totalMass;
        state.x  = state.x + tau * state.xDot;
        state.xDot = state.xDot + tau * xAcc;
        state.theta = state.theta + tau * state.thetaDot;
        state.thetaDot = state.thetaDot + tau * thetaAcc;

        rof.finished =  state.x < -xThreshold || state.x > xThreshold || state.theta < -thetaThresholdRadians || state.theta > thetaThresholdRadians;
        rof.reward = rof.finished ? 0 : 1;
		rof.observation = getObservation(state);
		// System.out.println(rof.finished);

		return rof;
	}

	// Implementation for renderer, return true on success
	// If observation or g is null return true if implemented
	protected boolean draw(Graphics g,  Info state){
		if(g == null || state == null){
			return true;
		}

		CartPoleState rs = (CartPoleState)state;
		
		g.setColor(Color.BLACK);
		g.drawLine((int)(rs.x * 500) + 300, (int)300, (int)(rs.x * 500) + 300 + (int)(Math.cos(rs.theta + Math.PI/2) * length  * 300), (int)(300 - (Math.sin(rs.theta + Math.PI/2) * length  * 300)));
		g.fillOval((int)(rs.x * 500) + 300 - 10, (int)300 - 10, 20, 20);

		g.drawString(""+(int)(Math.random()*10), 100, 100);

		return true;
	}
}
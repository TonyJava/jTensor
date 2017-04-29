package jREC;

import java.awt.Graphics;
import javax.swing.JFrame;
import javax.swing.JPanel;

public abstract class Environment{

	private JFrame jFrame = null;
	private boolean renderGraphics = true;

	protected Info state;

	// Returns amount/type of info available
	public abstract Space getObservationSpace();

	// Returns number of actions available
	public abstract Space getActionSpace();

	// Resets the environment, returns initial observation
	public abstract Info reset();

	// Returns initialized action object
	public abstract Info createAction();

	// Make action, return reward, observation, and finished flag
	public ROF step(Info action){
		ROF rof = nextState(state, action);
		state = rof.nextState;
		rof.nextState = null;
		if(rof.finished){
			sketchVar = 0;
		}
		return rof;
	}

	static int sketchVar = 0;

	// Renders the environment
	public void render(){
		if(renderGraphics && jFrame == null){
			if(draw(null, null)){
				jFrame = new JFrame("Environment");
				jFrame.add(new JPanel(){
					@Override
					public void paintComponent(Graphics g){
						draw(g, state);
						g.drawString(""+sketchVar++, 50, 50);
						g.drawString(""+System.currentTimeMillis(), 100, 50);
					}
				});
				jFrame.setSize(600, 600);
				jFrame.setVisible(true);
				jFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
			}else{
				renderGraphics = false;
			}
		}
		if(renderGraphics){
			jFrame.repaint();
		}else{
			renderString(state);
		}
	}

	// // Helper function for text based environments.
	// // Use by overriding render, implementing renderString(Info) 
	// // and calling this function.
	// protected void renderString(){
	// 	renderString(state);
	// }

	// Implementation for rendering env as a string
	// Used for console (or any non-JFrame) output
	// Will be called if draw(null, null) returns false
	protected void renderString(Info state){

	}

	// Implementation for renderer, return true on success
	// If observation or g is null return true if implemented
	protected boolean draw(Graphics g,  Info observation){
		return false;
	}

	// Implementation for next state
	protected abstract ROF nextState(Info state, Info action);

}

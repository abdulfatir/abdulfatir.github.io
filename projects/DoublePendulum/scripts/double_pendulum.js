var theta1 = 3*Math.PI/4;
var theta2 = Math.PI/2;
var dtheta1 = 0;
var dtheta2 = 0;
var d2theta1 = 0;
var d2theta2 = 0;
var X0 = 250;
var Y0 = 150;
var g = 15;
var m1 = 10;
var m2 = 10;
var l1 = 80;
var l2 = 80;
var dtime = 0.05;

var cos = Math.cos;
var sin = Math.sin;

var canvas = document.getElementById("pend-canvas");
var ctx = canvas.getContext("2d");
var canvas2 = document.getElementById("trail-canvas");
var ctx2 = canvas2.getContext("2d");
var init = {};

run();

function run()
{
	var thread1 = {x1:X0, y1:Y0, x2:0, y2:0};
	var thread2 = {x1:0, y1:0, x2:0, y2:0};
	var bob1 = {x:X0+l1*Math.sin(theta1), y:Y0+l1*Math.cos(theta1), lastx:0, lasty:0, mass:m1};
	var bob2 = {x:X0+l1*Math.sin(theta1)+l2*Math.sin(theta2), y:Y0+l1*Math.cos(theta1)+l2*Math.cos(theta2), lastx:0, lasty:0, mass:m2};

	clearInterval(init);
	ctx.clearRect(0,0, canvas.width, canvas.height);
	ctx2.clearRect(0,0, canvas2.width, canvas2.height);
	init = setInterval(function()
		{
			render(thread1, thread2, bob1, bob2, ctx, canvas);
		}, 10);
}

function render(thread1, thread2, bob1, bob2, context, canvas)
{
	mu = 1+m1/m2;
	d2theta1 = (g*(sin(theta2)*cos(theta1-theta2)-mu*sin(theta1)) - (l2*dtheta2*dtheta2+l1*dtheta1*dtheta1*cos(theta1-theta2))*sin(theta1-theta2))/(l1*(mu-cos(theta1-theta2)*cos(theta1-theta2)));
	d2theta2 = (g*mu*(sin(theta1)*cos(theta1-theta2)-sin(theta2)) + (mu*l1*dtheta1*dtheta1+l2*dtheta2*dtheta2*cos(theta1-theta2))*sin(theta1-theta2))/(l2*(mu-cos(theta1-theta2)*cos(theta1-theta2)));
	dtheta1 += d2theta1*dtime;
	dtheta2 += d2theta2*dtime;
	theta1 += dtheta1*dtime;
	theta2 += dtheta2*dtime;

	bob1.lastx = bob1.x;
	bob1.lasty = bob1.y;
	bob2.lastx = bob2.x;
	bob2.lasty = bob2.y;

	bob1.x = X0+l1*Math.sin(theta1);
	bob1.y = Y0+l1*Math.cos(theta1);
	bob2.x = X0+l1*Math.sin(theta1)+l2*Math.sin(theta2);
	bob2.y = Y0+l1*Math.cos(theta1)+l2*Math.cos(theta2);

	thread1.x2 = bob1.x;
	thread1.y2 = bob1.y;
	thread2.x1 = bob1.x;
	thread2.y1 = bob1.y;
	thread2.x2 = bob2.x;
	thread2.y2 = bob2.y;

	context.clearRect(0,0, canvas.width, canvas.height);

	drawSupport(context);
	drawThread(thread1, context);
	drawThread(thread2, context);
	drawBob(bob1, context);
	drawBob(bob2, context);
	drawTrail(bob1, bob2, context);
}

function drawSupport(context)
{
	context.beginPath();
	context.arc(X0, Y0, 3, 0, 2*Math.PI);
	context.fillStyle = "#00f";
	context.fill();
	context.lineWidth=1;
	context.stroke();

}

function drawThread(thread, context)
{
	context.beginPath();
	context.moveTo(thread.x1, thread.y1);
	context.lineTo(thread.x2, thread.y2);
	context.strokeStyle = "#000";
	context.lineWidth=3;
	context.stroke();
}

function drawBob(bob, context)
{
	context.beginPath();
	context.arc(bob.x, bob.y, bob.mass, 0, 2*Math.PI);
	context.fillStyle = "#ff0000";
	context.fill();
	context.lineWidth=1;
	context.stroke();
}

function drawTrail(bob1, bob2)
{
	ctx2.beginPath();
	ctx2.moveTo(bob2.lastx, bob2.lasty);
	ctx2.lineTo(bob2.x, bob2.y);
	ctx2.strokeStyle = "#00f";
	ctx2.lineWidth = 2;
	ctx2.stroke();
}
$(document).ready(function()
	{
		$("#reset").click(function()
		{
			m1 = $("#m1").val();
			m2 = $("#m2").val();
			theta1 = $("#theta1").val()*Math.PI/180.0;
			theta2 = $("#theta2").val()*Math.PI/180.0;
			l1 = $("#l1").val();
			l2 = $("#l2").val();
			g = $("#g").val();

			dtheta1 = 0;
			dtheta2 = 0;
			d2theta1 = 0;
			d2theta2 = 0;
			run();
		});
	});
#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include<math.h>
#include <windows.h>
#include <GL/glut.h>
#include <vector>
#include <stdint.h>
#include "bitmap_image.hpp"
#define pi (2*acos(0.0))
using namespace std;
//........................................//

double viewangle = 80;
int screen_width = 500;
int screen_height = 500;
int drawaxes;
double angle;
int imagewidth = 768;
int recursion_level = 4;

class point
{
public:
	double x, y, z;
	point() {

	}
	point(double a, double b, double c) {
		x = a;
		y = b;
		z = c;
	}
	point operator+(point pt) {
		point t = point(x + pt.x, y + pt.y, z + pt.z);
		return t;
	}

	point operator-(point pt) {
		point t = point(x - pt.x, y - pt.y, z - pt.z);
		return t;
	}

	point operator*(double mul) {
		point t = point(x * mul, y * mul, z * mul);
		return t;
	}

	point operator/(double div) {
		point t = point(x / div, y / div, z / div);
		return t;
	}

};

double dot(point pt1, point pt2)
{
	return (pt1.x*pt2.x + pt1.y*pt2.y + pt1.z*pt2.z);
}

point pos(100, 100, 10);
point l(-1 / sqrt(2), -1 / sqrt(2), 0);
point r(-1 / sqrt(2), 1 / sqrt(2), 0);
point u(0, 0, 1);
vector<point> lights;
class Ray {

public:
	point start;
	point dir;

	Ray(point a, point b) {

		start = a;
		double div = sqrt(b.x*b.x + b.y*b.y + b.z*b.z);
		dir = b / div;
	}
	Ray() {

	}
};

class Object
{

protected:
	point reference_point;
	double height, width, length;
	int Shine;
	double color[3];
	double co_efficients[4];
	double eta = 1.0;
	double source_factor = 1.0;
public:
	Object() {}
	virtual void draw() {};
	void setColor(double a, double b, double c);
	void setShine(int s);
	void setCoEfficients(double a, double b, double c, double d);
	virtual double intersect(Ray* ray,double intersectColor[3],int depth) {
		return -1;
	}
	virtual double getIntersectingT(Ray* ray) {
		return -1;
	}
	virtual point getNormal(point intersectionPoint) = 0;
	point reflection(Ray *ray, point normal);
	point refraction(Ray *ray, point normal);

};

void Object::setColor(double a, double b, double c)
{

	color[0] = a;
	color[1] = b;
	color[2] = c;

}

void Object::setCoEfficients(double a, double b, double c, double d)
{

	co_efficients[0] = a;
	co_efficients[1] = b;
	co_efficients[2] = c;
	co_efficients[3] = d;
}

void Object::setShine(int s)
{

	Shine = s;
}

point Object::reflection(Ray *ray, point normal)
{
	point reflect = ray->dir - (normal * 2.0) *  dot(ray->dir, normal);
	double dib = sqrt(reflect.x*reflect.x + reflect.y*reflect.y + reflect.z*reflect.z);
	return (reflect / dib);
}

point Object::refraction(Ray *ray, point normal)
{
	double N_dot_I = dot(normal, ray->dir);
	double k = 1.0 - eta * eta * (1.0 - N_dot_I * N_dot_I);
	if (k < 0)
	{
		point refract(0.0, 0.0, 0.0);
		return refract;
	}
	else
	{
		point refract = (ray->dir *eta) - normal * (eta * N_dot_I + sqrt(k));
		double dib = sqrt(refract.x*refract.x + refract.y*refract.y + refract.z*refract.z);
		return (refract / dib);
	}
}

vector<Object*> objects;

class Sphere :public Object
{

public:
	Sphere(point Center, double Radius)
	{
		reference_point = Center;
		length = Radius;
	}

	void draw()
	{
		//write codes for drawing sphere
		int slices = 24, stacks = 20;
		point points[100][100];
		int i, j;
		double h, r;
		//generate points
		glPushMatrix();
		glTranslatef(reference_point.x, reference_point.y, reference_point.z);
		for (i = 0; i <= stacks; i++)
		{
			h = length * sin(((double)i / (double)stacks)*(pi / 2));
			r = length * cos(((double)i / (double)stacks)*(pi / 2));
			for (j = 0; j <= slices; j++)
			{
				points[i][j].x = r * cos(((double)j / (double)slices) * 2 * pi);
				points[i][j].y = r * sin(((double)j / (double)slices) * 2 * pi);
				points[i][j].z = h;
			}
		}
		//draw quads using generated points
		for (i = 0; i<stacks; i++)
		{
			glColor3f(color[0], color[1], color[2]);
			for (j = 0; j<slices; j++)
			{
				glBegin(GL_QUADS);
				{
					//upper hemisphere
					glVertex3f(points[i][j].x, points[i][j].y, points[i][j].z);
					glVertex3f(points[i][j + 1].x, points[i][j + 1].y, points[i][j + 1].z);
					glVertex3f(points[i + 1][j + 1].x, points[i + 1][j + 1].y, points[i + 1][j + 1].z);
					glVertex3f(points[i + 1][j].x, points[i + 1][j].y, points[i + 1][j].z);
					//lower hemisphere
					glVertex3f(points[i][j].x, points[i][j].y, -points[i][j].z);
					glVertex3f(points[i][j + 1].x, points[i][j + 1].y, -points[i][j + 1].z);
					glVertex3f(points[i + 1][j + 1].x, points[i + 1][j + 1].y, -points[i + 1][j + 1].z);
					glVertex3f(points[i + 1][j].x, points[i + 1][j].y, -points[i + 1][j].z);
				}
				glEnd();
			}
		}
		glPopMatrix();

	}

	point getNormal(point intersectionPoint)
	{
		point normal = intersectionPoint - reference_point;
		double div = sqrt(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
		return (normal / div);
	}

	double getIntersectingT(Ray* ray)
	{
		point Ro = ray->start - reference_point;
		double A = dot(ray->dir, ray->dir);
		double B = 2 * dot(ray->dir, Ro);
		double C = dot(Ro, Ro) - length * length;
		double D = B * B - 4 * A * C;
		if (D < 0) return -1;
		double t1 = (-B+ sqrt(D)) / (2.0 * A);
		double t2 = (-B - sqrt(D)) / (2.0 * A);

		if (t1 < t2)
		{
			return t1;
		}
		return t2;
	}

	double intersect(Ray *ray, double intersectColor[3],int depth)
	{
		double t = getIntersectingT(ray);
		if (t <= 0) return -1;
		if (depth == 0)return t;
		intersectColor[0] = color[0] * co_efficients[0];
		intersectColor[1] = color[1] * co_efficients[0];
		intersectColor[2] = color[2] * co_efficients[0];
		point intersectingPoint(ray->start + (ray->dir * t));
		point normal = getNormal(intersectingPoint);
		point reflect = reflection(ray, normal);
		point refract = refraction(ray, normal);

		for (int i = 0; i < lights.size(); i++)
		{
			point direction = lights[i] - intersectingPoint;
			double div = sqrt(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
			direction = (direction / div);
			point start = intersectingPoint + (direction * 1.0);
			Ray *L = new Ray(start, direction);
			int obscured = 0;
			for (int j = 0; j < objects.size(); j++)
			{
				double t_reflection = objects[j]->getIntersectingT(L);
				if (t_reflection > 0 && abs(t_reflection) <= div)
				{
					obscured = 1;
					break;
				}

			}
			if (obscured == 0)
			{
				double lambert = dot(L->dir, normal);
				double phong = pow(dot(reflect, ray->dir), Shine);
				if (lambert < 0)
				{
					lambert = 0.0;
				}
				if (phong < 0)
				{
					phong = 0.0;
				}
				for (int l = 0; l < 3; l++)
				{
					intersectColor[l] += source_factor * lambert* co_efficients[1] * color[l];
					intersectColor[l] += source_factor * phong * co_efficients[2] * color[l];

				}
			}
			if (depth < recursion_level)
			{
				start = intersectingPoint + (reflect * 1.0);
				Ray *reflectionRay = new Ray(start, reflect);
				int nearest = -1;
				double refleced_color[3];
				double t_min = 9999999.99;
				for (int k = 0; k<objects.size(); k++)
				{
					double t = objects.at(k)->getIntersectingT(reflectionRay);
					if (t <= 0)continue;
					if (t < t_min)
					{
						nearest = k;
						t_min = t;
					}
				}
				if (nearest != -1) {
					double t = objects.at(nearest)->intersect(reflectionRay, refleced_color, depth+1);
					if(t <= 0)
                    {
                        return -1;
                    }
					for (int k = 0; k < 3; k++)
					{
						intersectColor[k] += refleced_color[k] * co_efficients[3];
					}
				}

				start = intersectingPoint + (refract * 1.0);
				Ray *refractionRay = new Ray(start, refract);
				nearest = -1;
				double refracted_color[3];
				t_min = 9999999.99;
				for (int k = 0; k < objects.size(); k++) {
					double tr = objects.at(k)->getIntersectingT(refractionRay);
					if (tr <= 0) continue;
					if (tr <t_min)
					{
						nearest = k;
						t_min = tr;
					}
				}

				if (nearest != -1) {
					double ts = objects[nearest]->intersect(refractionRay, refracted_color, depth + 1);
					if(ts <= 0)return -1;
					for (int k = 0; k<3; k++) {
						intersectColor[k] += (refracted_color[k] * eta);
					}
				}
			}
			for (int q = 0; q < 3; q++)
			{
				if (intersectColor[q] > 1)intersectColor[q] = 1;
				if (intersectColor[q] < 0)intersectColor[q] = 0;
			}

		}

		return t;

	}
};


class Floor : public Object {
public:
	double texture_width, texture_height;
	bitmap_image floor_texture;
	Floor(double FloorWidth, double TileWidth, bitmap_image texture) {
		floor_texture = texture;
		texture_width = texture.width() / FloorWidth;
		texture_height = texture.height() / FloorWidth;
		point p(-FloorWidth / 2, -FloorWidth / 2, 0);
		reference_point = p;
		length = TileWidth;
	}
	void draw() {
		//write codes for drawing black and white floor
		int tiles_to_fit = int(2 * abs(reference_point.x) / length);
		for (int i = 0; i < tiles_to_fit; i++) {
			for (int j = 0; j < tiles_to_fit; j++) {
				if ( ((i + j) % 2) == 0) glColor3f(0, 0, 0);
				else glColor3f(1, 1, 1);
				glBegin(GL_QUADS);
				{
					glVertex3f(reference_point.x + length * j, reference_point.y + length * i , reference_point.z);
					glVertex3f(reference_point.x + length * j + length, reference_point.y + length * i, reference_point.z);
					glVertex3f(reference_point.x + length * j + length, reference_point.y + length * i + length , reference_point.z);
					glVertex3f(reference_point.x + length * j, reference_point.y + length * i + length, reference_point.z);
				}
				glEnd();
			}
		}
	}

	point getNormal(point intersectionPoint)
	{
		point normal(0, 0, 1);
		return normal;
	}

	double getIntersectingT(Ray* ray)
	{
		point normal = getNormal(reference_point);
		double t = (-1.0) * dot(normal, ray->start) / dot(normal, ray->dir);
		return t;
	}

	double intersect(Ray *ray, double intersectColor[3], int depth)
	{
		double t = getIntersectingT(ray);
		if (t <= 0)return -1;
		//setColor(color[0] * co_efficients[0], color[1] * co_efficients[0], color[2] * co_efficients[0]);
		point intersectingPoint(ray->start + (ray->dir * t));
		if (intersectingPoint.x < reference_point.x || intersectingPoint.x > ( (-1)*reference_point.x) || intersectingPoint.y < reference_point.y || intersectingPoint.y >((-1)*reference_point.y))
		{
			return -1;
		}
		int xc= ceil(intersectingPoint.x / length);
		int yc = ceil(intersectingPoint.y / length);
		if ((xc + yc) % 2 == 0) {
			color[0] = 0;
			color[1] = 0;
			color[2] = 0;
		}
		else {
			color[0] = 1;
			color[1] = 1;
			color[2] = 1;
		}
		unsigned char red, green, blue;
		int x = texture_width * (intersectingPoint.x + abs(reference_point.x));
		int y = texture_height * (intersectingPoint.y + abs(reference_point.y));
		floor_texture.get_pixel(x, y, red, green, blue);
		double colors[3];
        colors[0] = red;
		colors[1] = green;
		colors[2] = blue;
		for (int i = 0; i<3; i++)
        {
			intersectColor[i] = ( color[i] * co_efficients[0] * colors[i] )/ 255.0;
		}
		point normal = getNormal(intersectingPoint);
		point reflect = reflection(ray, normal);
		point refract = refraction(ray, normal);

		for (int i = 0; i < lights.size(); i++)
		{
			point direction = lights[i] - intersectingPoint;
			double div = sqrt(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
			direction = (direction / div);
			point start = intersectingPoint + (direction * 1.0);
			Ray *L = new Ray(start, direction);
			int obscured = 0;
			for (int j = 0; j < objects.size(); j++)
			{
				double tl = objects[j]->getIntersectingT(L);
				if (tl > 0)
				{
					obscured = 1;
					break;
				}
			}
			if (obscured == 0)
			{
				double lambert = dot(L->dir, normal);
				double phong = pow(dot(reflect, ray->dir), Shine);
				if (lambert < 0)
				{
					lambert = 0.0;
				}
				if (phong < 0)
				{
					phong = 0.0;
				}
				for (int l = 0; l < 3; l++)
				{
					intersectColor[l] += source_factor * lambert* co_efficients[1] * color[l];
					intersectColor[l] += source_factor * phong * co_efficients[2] * color[l];

				}
			}
			if (depth == 0)return t;
			if (depth < recursion_level)
			{
				start = intersectingPoint + reflect * 1.0;
				Ray *reflectionRay = new Ray(start, reflect);
				int nearest = -1;
				double refleced_color[3];
				double t_min = 9999.99;
				for (int k = 0; k<objects.size(); k++)
				{
					double t = objects.at(k)->getIntersectingT(reflectionRay);
					if (t <= 0)continue;
					if (t < t_min) {
						nearest = k;
						t_min = t;
					}
				}
				if (nearest != -1) {
					double t = objects.at(nearest)->intersect(reflectionRay, refleced_color, depth + 1);
					if(t <= 0)return -1;
					for (int k = 0; k < 3; k++)
					{
						intersectColor[k] += refleced_color[k] * co_efficients[3];
					}
				}
			}
			for (int q = 0; q < 3; q++)
			{
				if (intersectColor[q] > 1)intersectColor[q] = 1;
				if (intersectColor[q] < 0)intersectColor[q] = 0;
			}

		}

		return t;

	}
};


class Triangle : public Object {

	public:
		point a, b, c;
		Triangle()
		{

		}
		Triangle(point p1, point p2, point p3)
		{
			a = p1;
			b = p2;
			c = p3;
		}
		void draw()
		{
			double r, g, bl;
			r = color[0];
			g = color[1];
			bl = color[2];
			glColor3f(r, g, bl);
			glBegin(GL_TRIANGLES);
			{
				glVertex3f(a.x, a.y, a.z);
				glVertex3f(b.x, b.y, b.z);
				glVertex3f(c.x, c.y, c.z);
			}
			glEnd();
		}
		point cross(point p, point q)
		{
			point normal;
			normal.x = p.y * q.z - p.z * q.y;
			normal.y = p.z * q.x - p.x * q.z;
			normal.z = p.x * q.y - p.y * q.x;
			return normal;
		}
		point getNormal(point intersectingPoint)
		{
			point normal;
			point p = b - a;
			point q = c - a;
			normal = cross(p, q);
			double div = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
			return (normal / div);
		}

		double getIntersectingT(Ray* ray)
		{
			const double EPSILON = 0.0000001;
			point edge1, edge2, h, s, q;
			double al, f, u, v;
			edge1 = b - a;
			edge2 = c - a;
			h = cross(ray->dir, edge2);
			al = dot(edge1, h);
			if (al > -EPSILON && al < EPSILON)return -1;
			f = 1.0 / al;
			s = ray->start - a;
			u = f * (dot(s, h));
			if (u < 0.0 || u > 1.0)return -1;
			q = cross(s,edge1);
			v = f * (dot(ray->dir,q));
			if (v < 0.0 || u + v > 1.0)return -1;
			double t = f * dot(edge2,q);
			if (t > EPSILON) return t;
			else return -1;
		}

		double intersect(Ray *ray, double intersectColor[3], int depth)
		{
			double t = getIntersectingT(ray);
			if (t <= 0) return -1;
			if (depth == 0)return t;
			intersectColor[0] = color[0] * co_efficients[0];
			intersectColor[1] = color[1] * co_efficients[0];
			intersectColor[2] = color[2] * co_efficients[0];
			point intersectingPoint(ray->start + (ray->dir * t));

			for (int i = 0; i < lights.size(); i++)
			{
				point normal = getNormal(intersectingPoint);
				point reflect = reflection(ray, normal);
				point direction = lights[i] - intersectingPoint;
				double div = sqrt(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
				direction = (direction / div);
				if (dot(direction, normal) > 0) normal = normal * (-1);
				point start = intersectingPoint + (direction * 1.0);
				Ray *L = new Ray(start, direction);
				int obscured = 0;
				for (int j = 0; j < objects.size(); j++)
				{
					double t_reflection = objects[j]->getIntersectingT(L);
					if (t_reflection > 0 && abs(t_reflection) <= div)
					{
						obscured = 1;
						break;
					}

				}
				if (obscured == 0)
				{
					double lambert = dot(L->dir, normal);
					double phong = pow(dot(reflect, ray->dir), Shine);
					if (lambert < 0)
					{
						lambert = 0.0;
					}
					if (phong < 0)
					{
						phong = 0.0;
					}
					for (int l = 0; l < 3; l++)
					{
						intersectColor[l] += source_factor * lambert* co_efficients[1] * color[l];
						intersectColor[l] += source_factor * phong * co_efficients[2] * color[l];

					}
				}

				if (depth < recursion_level)
				{
					start = intersectingPoint + (reflect * 1.0);
					Ray *reflectionRay = new Ray(start, reflect);
					int nearest = -1;
					double refleced_color[3];
					double t_min = 9999999.99;
					for (int k = 0; k<objects.size(); k++)
					{
						double tt = objects.at(k)->getIntersectingT(reflectionRay);
						if (tt <= 0)continue;
						if (tt < t_min)
						{
							nearest = k;
							t_min = tt;
						}
					}
					double r[3];
					if (nearest != -1) {
						double tk = objects.at(nearest)->intersect(reflectionRay, r, depth + 1);
						if(tk<=0) return -1;
						for (int k = 0; k < 3; k++)
						{
							intersectColor[k] += r[k] * co_efficients[3];
						}
					}

				}
				for (int q = 0; q < 3; q++)
				{
					if (intersectColor[q] > 1)intersectColor[q] = 1;
					if (intersectColor[q] < 0)intersectColor[q] = 0;
				}

			}

			return t;
		}

};

class Quadrics : public Object {

public:
	double A, B, C, D, E, F, G, H, I, J;

	Quadrics(double a, double b, double c, double d, double e, double f, double g, double h, double i, double j, point r, double l, double w, double hi)
	{
		A = a;
		B = b;
		C = c;
		D = d;
		E = e;
		F = f;
		G = g;
		H = h;
		I = i;
		J = j;
		reference_point = r;
		length = l;
		width = w;
		height = hi;
	}
	Quadrics() {

	}
	void draw()
	{

	}

	point getNormal(point intersectingPoint)
	{
		point normal(2 * A * intersectingPoint.x + D * intersectingPoint.y + E * intersectingPoint.z + G,
			D * intersectingPoint.x + 2 * B * intersectingPoint.y + F * intersectingPoint.z + H,
			E * intersectingPoint.x + F * intersectingPoint.y + 2 * C * intersectingPoint.z + I);
		double div = sqrt(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
		return (normal / div);
	}

	double getIntersectingT(Ray* ray)
	{
		double x0 = ray->start.x;
		double y0 = ray->start.y;
		double z0 = ray->start.z;
		double x1 = ray->dir.x;
		double y1 = ray->dir.y;
		double z1 = ray->dir.z;
		double a = A * x1 * x1 + B * y1 * y1 + C * z1 * z1 + D * x1 * y1 + E * x1 * z1 + F * y1 * z1 ;
		double b = 2 * x0 * x1 * A + 2 * y0 * y1 *B + 2 * z0 * z1* C + D * x0 * y1 + D * x1 * y0; +E * x0* z1 + E * x1 * z0 + F * y0 * z1 + F * y1 * z0
				   + G* x1 + H * y1 + I * z1;
		double c = A * x0 * x0 + B * y0 * y0 + C * z0 * z0 + D * x0 * y0 + E * x0 * z0 + F * z0 * y0 + G * x0 + H * y0 + I * z0 + J;
		double d = b * b - 4 * a*c;
		if (d < 0) return -1;
		double t1 = (-b + sqrt(d)) / (2.0*a);
		double t2 = (-b - sqrt(d)) / (2.0*a);
		point intp1 = ray->start + ray->dir * t1;
		point intp2 = ray->start + ray->dir * t2;
		bool noIntersect1 = (length > 0 && (intp1.x < reference_point.x || intp1.x > (reference_point.x + length)) ||width > 0 && (intp1.y < reference_point.y || intp1.y > (reference_point.y + width)) ||height > 0 && (intp1.z < reference_point.z || intp1.z > (reference_point.z + height)));
		bool noIntersect2 = (length > 0 && (intp2.x < reference_point.x || intp2.x >(reference_point.x + length)) ||width > 0 && (intp2.y < reference_point.y || intp2.y >(reference_point.y + width)) ||height > 0 && (intp2.z < reference_point.z || intp2.z >(reference_point.z + height)));
		if (noIntersect1 && noIntersect2)return -1;
		if (!noIntersect1 && noIntersect2)return t1;
		else if (noIntersect1 && !noIntersect2)return t2;
		else {
			if (t1 < t2)
			{
				return t1;
			}
			return t2;
		}
	}

	double intersect(Ray *ray, double intersectColor[3], int depth)
	{
		double t = getIntersectingT(ray);
		if (t <= 0)return -1;
		if (depth == 0)return t;
		//setColor(color[0] * co_efficients[0], color[1] * co_efficients[0], color[2] * co_efficients[0]);
		intersectColor[0] = color[0] * co_efficients[0];
		intersectColor[1] = color[1] * co_efficients[0];
		intersectColor[2] = color[2] * co_efficients[0];
		point intersectingPoint(ray->start + (ray->dir * t));
		point normal = getNormal(intersectingPoint);
		point reflect = reflection(ray, normal);
		point refract = refraction(ray, normal);

		for (int i = 0; i < lights.size(); i++)
		{
			point direction = lights[i] - intersectingPoint;
			double div = sqrt(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
			direction = (direction / div);
			point start = intersectingPoint + (direction * 1.0);
			Ray *L = new Ray(start, direction);
			int obscured = 0;
			for (int j = 0; j < objects.size(); j++)
			{
				double tl = objects[j]->getIntersectingT(L);
				if (tl > 0 && abs(tl) < div)
				{
					obscured = 1;
					break;
				}

			}
			if (obscured == 0)
			{
				double lambert = dot(L->dir, normal);
				double phong = pow(dot(reflect, ray->dir), Shine);
				if (lambert < 0)
				{
					lambert = 0.0;
				}
				if (phong < 0)
				{
					phong = 0.0;
				}
				for (int l = 0; l < 3; l++)
				{
					intersectColor[l] += source_factor * lambert* co_efficients[1] * color[l];
					intersectColor[l] += source_factor * pow(phong, Shine) * co_efficients[2] * color[l];

				}
			}
			if (depth < recursion_level)
			{
				start = intersectingPoint + reflect * 1.0;
				Ray *reflectionRay = new Ray(start, reflect);
				int nearest = -1;
				double refleced_color[3];
				double t_min = 9999.99;
				for (int k = 0; k<objects.size(); k++)
				{
					double t = objects.at(k)->getIntersectingT(reflectionRay);
					if (t <= 0)continue;
					if (t < t_min) {
						nearest = k;
						t_min = t;
					}
				}
				if (nearest != -1) {
					double t = objects.at(nearest)->intersect(reflectionRay, refleced_color, depth + 1);
					if(t<=0)return -1;
					for (int k = 0; k < 3; k++)
					{
						intersectColor[k] += refleced_color[k] * co_efficients[3];
					}
				}
			}
		}
		return t;

	}
};


//........................................//




void normalize(point p)
{
	double normal = sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
	p.x /= normal;
	p.y /= normal;
	p.z /= normal;
}

void drawAxes()
{
	if (drawaxes == 1)
	{
		glColor3f(1.0, 1.0, 1.0);
		glBegin(GL_LINES); {
			glVertex3f(100, 0, 0);
			glVertex3f(-100, 0, 0);

			glVertex3f(0, -100, 0);
			glVertex3f(0, 100, 0);

			glVertex3f(0, 0, 100);
			glVertex3f(0, 0, -100);
		}glEnd();
	}
}


void capture() {

	bitmap_image image(imagewidth, imagewidth);
	for (int i = 0; i < imagewidth; i++)
	{
		for (int j = 0; j < imagewidth; j++)
		{
			image.set_pixel(j, i, 0, 0, 0);
		}
	}
	double plane_distance = (screen_height / 2) / (tan(viewangle * pi / 360));
	point top_left = pos + (l * plane_distance - r * (screen_width / 2) + u * (screen_height / 2));

	double du = (screen_width * 1.0) / imagewidth;
	double dv = (screen_height * 1.0) / imagewidth;
	for (int i = 0; i < imagewidth; i++) {
		for (int j = 0; j < imagewidth; j++) {
			point corner, dir;
			corner = top_left + r * j * du - u * i * dv;
			dir = corner - pos;
			Ray *ray = new Ray(pos, dir);
			int nearest = -1;
			double dummyColorAt[3];
			double t_min = 9999999.99;
			for (int k = 0; k<objects.size(); k++)
			{
				double t = objects.at(k)->intersect(ray, dummyColorAt, 0);
				//dummyColorAt is the color array where pixel value will be stored in return time.
				//As this is only for nearest object detection dummy should be sufficient. Level is 0 here

				if (t <= 0)continue;
				if (t < t_min)
				{
					nearest = k;
					t_min = t;
				}
			}
			double colorAt[3];
			if (nearest != -1)
			{
				double t= objects.at(nearest)->intersect(ray, colorAt, 1);
				image.set_pixel(j, i, int(255 * colorAt[0]), int(255 * colorAt[1]), int(255 * colorAt[2]));
			}
		}
	}
	image.save_image("1.bmp");
    objects.clear();
    lights.clear();
}


void keyboardListener(unsigned char key, int x, int y) {
	point n_lr(0, 0, 0);
	point n_ur(0, 0, 0);
	point n_lu(0, 0, 0);
	point n_ru(0, 0, 0);
	point n_ul(0, 0, 0);
	point n_rl(0, 0, 0);
	switch (key) {
		double x, y, z;

	case '5':

		n_ul.x = l.y*u.z - l.z*u.y;
		n_ul.y = l.z*u.x - l.x*u.z;
		n_ul.z = l.x*u.y - l.y*u.x;

		n_rl.x = l.y*r.z - l.z*r.y;
		n_rl.y = l.z*r.x - l.x*r.z;
		n_rl.z = l.x*r.y - l.y*r.x;

		x = u.x * cos(3 * pi / 180) + n_ul.x * sin(3 * pi / 180);
		y = u.y * cos(3 * pi / 180) + n_ul.y * sin(3 * pi / 180);
		z = u.z * cos(3 * pi / 180) + n_ul.z * sin(3 * pi / 180);

		r.x = r.x * cos(3 * pi / 180) + n_rl.x * sin(3 * pi / 180);
		r.y = r.y * cos(3 * pi / 180) + n_rl.y * sin(3 * pi / 180);
		r.z = r.z * cos(3 * pi / 180) + n_rl.z * sin(3 * pi / 180);
		u.x = x;
		u.y = y;
		u.z = z;

		normalize(r);
		normalize(u);

		break;
	case '6':

		n_ul.x = l.y*u.z - l.z*u.y;
		n_ul.y = l.z*u.x - l.x*u.z;
		n_ul.z = l.x*u.y - l.y*u.x;

		n_rl.x = l.y*r.z - l.z*r.y;
		n_rl.y = l.z*r.x - l.x*r.z;
		n_rl.z = l.x*r.y - l.y*r.x;

		x = u.x * cos(-3 * pi / 180) + n_ul.x * sin(-3 * pi / 180);
		y = u.y * cos(-3 * pi / 180) + n_ul.y * sin(-3 * pi / 180);
		z = u.z * cos(-3 * pi / 180) + n_ul.z * sin(-3 * pi / 180);

		r.x = r.x * cos(-3 * pi / 180) + n_rl.x * sin(-3 * pi / 180);
		r.y = r.y * cos(-3 * pi / 180) + n_rl.y * sin(-3 * pi / 180);
		r.z = r.z * cos(-3 * pi / 180) + n_rl.z * sin(-3 * pi / 180);
		u.x = x;
		u.y = y;
		u.z = z;

		normalize(r);
		normalize(u);
		break;
	case '3':

		n_lr.x = r.y*l.z - r.z*l.y;
		n_lr.y = r.z*l.x - r.x*l.z;
		n_lr.z = r.x*l.y - r.y*l.x;

		n_ur.x = r.y*u.z - r.z*u.y;
		n_ur.y = r.z*u.x - r.x*u.z;
		n_ur.z = r.x*u.y - r.y*u.x;

		x = l.x * cos(3 * pi / 180) + n_lr.x * sin(3 * pi / 180);
		y = l.y * cos(3 * pi / 180) + n_lr.y * sin(3 * pi / 180);
		z = l.z * cos(3 * pi / 180) + n_lr.z * sin(3 * pi / 180);

		u.x = u.x * cos(3 * pi / 180) + n_ur.x * sin(3 * pi / 180);
		u.y = u.y * cos(3 * pi / 180) + n_ur.y * sin(3 * pi / 180);
		u.z = u.z * cos(3 * pi / 180) + n_ur.z * sin(3 * pi / 180);
		l.x = x;
		l.y = y;
		l.z = z;

		normalize(l);
		normalize(u);
		break;

	case '4':

		n_lr.x = r.y*l.z - r.z*l.y;
		n_lr.y = r.z*l.x - r.x*l.z;
		n_lr.z = r.x*l.y - r.y*l.x;

		n_ur.x = r.y*u.z - r.z*u.y;
		n_ur.y = r.z*u.x - r.x*u.z;
		n_ur.z = r.x*u.y - r.y*u.x;

		x = l.x * cos(-3 * pi / 180) + n_lr.x * sin(-3 * pi / 180);
		y = l.y * cos(-3 * pi / 180) + n_lr.y * sin(-3 * pi / 180);
		z = l.z * cos(-3 * pi / 180) + n_lr.z * sin(-3 * pi / 180);

		u.x = u.x * cos(-3 * pi / 180) + n_ur.x * sin(-3 * pi / 180);
		u.y = u.y * cos(-3 * pi / 180) + n_ur.y * sin(-3 * pi / 180);
		u.z = u.z * cos(-3 * pi / 180) + n_ur.z * sin(-3 * pi / 180);
		l.x = x;
		l.y = y;
		l.z = z;

		normalize(l);
		normalize(u);
		break;
	case '1':

		n_lu.x = u.y*l.z - u.z*l.y;
		n_lu.y = u.z*l.x - u.x*l.z;
		n_lu.z = u.x*l.y - u.y*l.x;

		n_ru.x = u.y*r.z - u.z*r.y;
		n_ru.y = u.z*r.x - u.x*r.z;
		n_ru.z = u.x*r.y - u.y*r.x;

		x = l.x * cos(3 * pi / 180) + n_lu.x * sin(3 * pi / 180);
		y = l.y * cos(3 * pi / 180) + n_lu.y * sin(3 * pi / 180);
		z = l.z * cos(3 * pi / 180) + n_lu.z * sin(3 * pi / 180);

		r.x = r.x * cos(3 * pi / 180) + n_ru.x * sin(3 * pi / 180);
		r.y = r.y * cos(3 * pi / 180) + n_ru.y * sin(3 * pi / 180);
		r.z = r.z * cos(3 * pi / 180) + n_ru.z * sin(3 * pi / 180);
		l.x = x;
		l.y = y;
		l.z = z;

		normalize(l);
		normalize(u);
		break;
	case '2':
		n_lu.x = u.y*l.z - u.z*l.y;
		n_lu.y = u.z*l.x - u.x*l.z;
		n_lu.z = u.x*l.y - u.y*l.x;

		n_ru.x = u.y*r.z - u.z*r.y;
		n_ru.y = u.z*r.x - u.x*r.z;
		n_ru.z = u.x*r.y - u.y*r.x;

		x = l.x * cos(-3 * pi / 180) + n_lu.x * sin(-3 * pi / 180);
		y = l.y * cos(-3 * pi / 180) + n_lu.y * sin(-3 * pi / 180);
		z = l.z * cos(-3 * pi / 180) + n_lu.z * sin(-3 * pi / 180);

		r.x = r.x * cos(-3 * pi / 180) + n_ru.x * sin(-3 * pi / 180);
		r.y = r.y * cos(-3 * pi / 180) + n_ru.y * sin(-3 * pi / 180);
		r.z = r.z * cos(-3 * pi / 180) + n_ru.z * sin(-3 * pi / 180);
		l.x = x;
		l.y = y;
		l.z = z;

		normalize(l);
		normalize(u);
		break;

	case '0':
		capture();
		break;

	default:
		break;
	}
}


void specialKeyListener(int key, int x, int y) {
	switch (key) {
	case GLUT_KEY_DOWN:		//down arrow key
		pos.x -= 2 * l.x;
		pos.y -= 2 * l.y;
		pos.z -= 2 * l.z;
		break;
	case GLUT_KEY_UP:		// up arrow key
		pos.x += 2 * l.x;
		pos.y += 2 * l.y;
		pos.z += 2 * l.z;
		break;

	case GLUT_KEY_RIGHT:
		pos.x += 2 * r.x;
		pos.y += 2 * r.y;
		pos.z += 2 * r.z;

		break;
	case GLUT_KEY_LEFT:
		pos.x -= 2 * r.x;
		pos.y -= 2 * r.y;
		pos.z -= 2 * r.z;
		break;

	case GLUT_KEY_PAGE_UP:
		pos.x -= 2 * u.x;
		pos.y -= 2 * u.y;
		pos.z -= 2 * u.z;
		break;

	case GLUT_KEY_PAGE_DOWN:
		pos.x += 2 * u.x;
		pos.y += 2 * u.y;
		pos.z += 2 * u.z;
		break;

	case GLUT_KEY_INSERT:
		break;

	case GLUT_KEY_HOME:

		break;
	case GLUT_KEY_END:

		break;

	default:
		break;
	}
}


void mouseListener(int button, int state, int x, int y) {	//x, y is the x-y of the screen (2D)
	switch (button) {
	case GLUT_LEFT_BUTTON:
		if (state == GLUT_DOWN) {		// 2 times?? in ONE click? -- solution is checking DOWN or UP
			drawaxes = 1 - drawaxes;
		}
		break;

	case GLUT_RIGHT_BUTTON:
		//........
		break;

	case GLUT_MIDDLE_BUTTON:
		//........
		break;

	default:
		break;
	}
}



void display() {

	//clear the display
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(0, 0, 0, 0);	//color black
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	/********************
	/ set-up camera here
	********************/
	//load the correct matrix -- MODEL-VIEW matrix
	glMatrixMode(GL_MODELVIEW);

	//initialize the matrix
	glLoadIdentity();

	//now give three info
	//1. where is the camera (viewer)?
	//2. where is the camera looking?
	//3. Which direction is the camera's UP direction?

	//gluLookAt(100,100,100,	0,0,0,	0,0,1);
	gluLookAt(pos.x, pos.y, pos.z, pos.x + l.x, pos.y + l.y, pos.z + l.z, u.x, u.y, u.z);


	//again select MODEL-VIEW
	glMatrixMode(GL_MODELVIEW);


	/****************************
	/ Add your objects from here
	****************************/
	//add objects

	drawAxes();
	/*
	for(vector<Object>::iterator it = objects.begin(); it != objects.end(); ++it) {
	it->

	}
	*/
	for (int i = 0; i<objects.size(); i++)
	{
		objects.at(i)->draw();
	}
	glBegin(GL_POINTS);
	for (vector<point>::iterator it = lights.begin(); it != lights.end(); ++it) {

		glColor3f(0.3, 0.3, 0.3);
		glPointSize(5.0f);  // wat
		glVertex3d(it->x, it->y, it->z);

	}
	glEnd();
	//ADD this line in the end --- if you use double buffer (i.e. GL_DOUBLE)
	glutSwapBuffers();
}


void animate() {
	//codes for any changes in Models, Camera
	glutPostRedisplay();
}

void init() {
	//codes for initialization
	drawaxes = 1;
	angle = 0;

	//clear the screen
	glClearColor(0, 0, 0, 0);

	/************************
	/ set-up projection here
	************************/
	//load the PROJECTION matrix
	glMatrixMode(GL_PROJECTION);

	//initialize the matrix
	glLoadIdentity();

	//give PERSPECTIVE parameters
	gluPerspective(viewangle, 1, 1, 1000.0);
	//field of view in the Y (vertically)
	//aspect ratio that determines the field of view in the X direction (horizontally)
	//near distance
	//far distance
}


void loadTestData()
{

	point Center(40, 0, 10);
	Sphere* sph = new Sphere(Center, 10);
	Object* obj = sph;
	obj->setColor(0, 1, 0);
	obj->setCoEfficients(0.4, 0.2, 0.2, 0.2);
	obj->setShine(10);
	objects.push_back(obj);

	point Center1(-30, 60, 20);
	Sphere* sph1 = new Sphere(Center1, 20);
	Object* obj1 = sph1;
	obj1->setColor(0, 0, 1);
	obj1->setCoEfficients(0.2, 0.2, 0.4, 0.2);
	obj1->setShine(15);
	objects.push_back(obj1);

	point Center2(-15, 15, 45);
	Sphere* sph2 = new Sphere(Center2, 15);
	Object* obj2 = sph2;
	obj2->setColor(1, 1, 0);
	obj2->setCoEfficients(0.4, 0.3, 0.1, 0.2);
	obj2->setShine(5);
	objects.push_back(obj2);


	point light1 = point(-70, 70, 70);
	lights.push_back(light1);
	point light2 = point(70, 70, 70);
	lights.push_back(light2);

	bitmap_image img("bd.bmp");
	Floor *floor = new Floor(1000, 20, img);
	Object* obj3 = floor;
	obj3->setCoEfficients(0.4, 0.2, 0.2, 0.2);
	obj3->setShine(1);
	objects.push_back(obj3);

	point reference(0, 0, 0);
	Object *temp;
	temp = new Quadrics(1, 1, 1, 0, 0, 0, 0, 0, 0, -100, reference, 0, 0, 20);
	temp->setColor(0, 1, 0);
	temp->setCoEfficients(0.4, 0.2, 0.1, 0.3);
	temp->setShine(10);
	objects.push_back(temp);

	point reference1(0, 0, 0);
	Object *temp1;
	temp1 = new Quadrics(0.0625, 0.04, 0.04, 0, 0, 0, 0, 0, 0, -36, reference, 0, 0, 15);
	temp1->setColor(1, 0, 0);
	temp1->setCoEfficients(0.4, 0.2, 0.1, 0.3);
	temp1->setShine(15);
	objects.push_back(temp1);

}

void loadFromFile()
{
	int no_of_object;
	Object* obj;
	FILE *f;
	f = freopen("scene.txt", "r", stdin);
	if (f == NULL)
	{
		printf("No File");
	}
	cin >> recursion_level;
	cin >> imagewidth;
	cin >> no_of_object;
	for (int i = 0; i < no_of_object; i++)
	{
		string objc;
		cin >> objc;
		//cout << objc<<"\n";
		if (objc == "sphere")
		{
			double x, y, z, r;
			cin >> x >> y >> z;
			point Center(x, y, z);
			cin >> r;
			Sphere* sph = new Sphere(Center, r);
			obj = sph;
			cin >> x >> y >> z;
			obj->setColor(x, y, z);
			cin >> x >> y >> z >> r;
			obj->setCoEfficients(x, y, z, r);
			cin >> x;
			obj->setShine(x);
			objects.push_back(obj);

		}
		else if (objc == "general")
		{
			double A, B, C, D, E, F, G, H, I, J, x, y, z, w, u, v ;
			cin >> A >> B >> C >> D >> E >> F >> G >> H >> I >> J;
			cin >> x >> y >> z >> w >> u >> v;
			point reference(x, y, z);
			Quadrics* quad = new Quadrics(A, B, C, D, E, F, G, H, I, J, reference, w, u, v);
			obj = quad;
			cin >> x >> y >> z;
			obj->setColor(x, y, z);
			cin >> x >> y >> z >> w;
			obj->setCoEfficients(x, y, z, w);
			cin >> x;
			obj->setShine(x);
			objects.push_back(obj);
		}
		else if (objc == "triangle")
		{

			double x, y, z, w;
			cin >> x >> y >> z;
			point a(x, y, z);
			cin >> x >> y >> z;
			point b(x, y, z);
			cin >> x >> y >> z;
			point c(x, y, z);
			Triangle* tri = new Triangle(a, b, c);
			obj = tri;
			cin >> x >> y >> z;
			obj->setColor(x, y, z);
			cin >> x >> y >> z >> w;
			obj->setCoEfficients(x, y, z, w);
			cin >> x;
			obj->setShine(x);
			objects.push_back(obj);

		}
	}

	bitmap_image img("2.bmp");
	Floor *floor = new Floor(1000, 20, img);
	obj = floor;
	obj->setCoEfficients(0.4, 0.2, 0.2, 0.2);
	obj->setShine(1);
	objects.push_back(obj);

	cin >> no_of_object;
	for (int i = 0; i < no_of_object; i++)
	{
		double x, y, z;
		cin >> x >> y >> z;
		point light = point(x, y, z);
		lights.push_back(light);
	}
	fclose(f);
}


int main(int argc, char **argv) {
	//loadTestData();
	loadFromFile();
	glutInit(&argc, argv);
	glutInitWindowSize(screen_width, screen_height);
	glutInitWindowPosition(0, 0);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);	//Depth, Double buffer, RGB color

	glutCreateWindow("My OpenGL Program");

	init();
	glEnable(GL_DEPTH_TEST);	//enable Depth Testing

	glutDisplayFunc(display);	//display callback function
	glutIdleFunc(animate);		//what you want to do in the idle time (when no drawing is occuring)

	glutKeyboardFunc(keyboardListener);
	glutSpecialFunc(specialKeyListener);
	glutMouseFunc(mouseListener);

	glutMainLoop();		//The main loop of OpenGL

	return 0;
}

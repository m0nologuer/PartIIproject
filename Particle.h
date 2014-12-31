
struct Particle{
	struct vec3{
		double x;
		double y;
		double z;
		vec3(){};
		vec3(double _x, double _y, double _z) :x(_x), y(_y), z(_z) {};

		vec3 operator*(double f)
		{
			vec3 result;
			result.x = this->x *f;
			result.y = this->y *f;
			result.z = this->z *f;
			return result;
		}
		vec3 operator+(vec3& y)
		{
			vec3 result;
			result.x = this->x + y.x;
			result.y = this->y + y.y;
			result.z = this->z + y.z;
			return result;
		}
		vec3 operator-(vec3& y)
		{
			vec3 result;
			result.x = this->x - y.x;
			result.y = this->y - y.y;
			result.z = this->z - y.z;
			return result;
		}

		static double dot(vec3 x, vec3 y)
		{
			return x.x*y.x + x.y*y.y + x.z*y.z;
		}
	};

	vec3 pos, speed, predicted_pos, d_p_pos;
	unsigned char r, g, b, a; // Color
	double lambda;
	double size, angle, weight;
	double life; // Remaining life of the particle. if < 0 : dead and unused.
	
	double cameradistance; //for sorting
	bool operator<(Particle const& p)
	{
		if (p.life < 0) //dead particles go at the end
			return false;
		else
			return (cameradistance < p.cameradistance);
	}
};
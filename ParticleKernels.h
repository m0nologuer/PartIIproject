double ParticleContainer::W_spiky(Particle::vec3 r)
{
	double radius = r.x*r.x + r.y*r.y + r.z*r.z;

	if (radius < h_squared)
	{
		//constant is 15/pi
		double result = pow(h - sqrt(radius), 3.0) * 4.77464829 / h_6;
		return result;
	}
	else //ignore particles outside a certain large radius
		return 0;
}
double ParticleContainer::W_poly6(Particle::vec3 r){

	double radius = r.x*r.x + r.y*r.y + r.z*r.z;

	if (radius < h_squared)
	{
		//constant is 315/64pi
		double result = pow(h_squared - radius, 3.0) * 1.56668147 / h_9;
		return result;
	}
	else //ignore particles outside a certain large radius
		return 0;
}
Particle::vec3 ParticleContainer::dW_spiky(Particle::vec3 r){

	double radius = r.x*r.x + r.y*r.y + r.z*r.z;

	if (radius < h_squared)
	{
		//constant is 15/pi
		double result = -3 * pow(h - sqrt(radius), 2.0) * 4.77464829 / h_6;
		Particle::vec3 grad = r * result;
		return grad;
	}
	else //ignore particles outside a certain large radius
		return Particle::vec3(0, 0, 0);
}
Particle::vec3 ParticleContainer::dW_poly6(Particle::vec3 r){

	double radius_2 = r.x*r.x + r.y*r.y + r.z*r.z;

	if (radius_2 < h_squared)
	{
		//constant is 315/64pi
		double radius = sqrt(radius_2);
		double result = -6 * radius * pow(h_squared - radius_2, 2.0) * 1.56668147 / h_9;
		Particle::vec3 grad = r * result;
		return grad;
	}
	else //ignore particles outside a certain large radius
		return Particle::vec3(0, 0, 0);
}
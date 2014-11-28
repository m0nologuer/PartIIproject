#include "ParticleContainer.h"
#include "../util/shader.hpp"
#include "../util/texture.hpp"

void ParticleContainer::Init(char* texture, char* vertexShader, char* pixelShader)
{
	//initalized matrices
	glGetFloatv(GL_MODELVIEW_MATRIX, (GLfloat*)&viewMat);
	glGetFloatv(GL_PROJECTION_MATRIX, (GLfloat*)&projMat);


	//load textures and shaders
	programID = LoadShaders(vertexShader, pixelShader);
	textureID = glGetUniformLocation(programID, "myTextureSampler");
	viewVec1ID =  glGetUniformLocation(programID, "CameraRight_worldspace");
	viewVec2ID = glGetUniformLocation(programID, "CameraUp_worldspace");
	vpMatID = glGetUniformLocation(programID, "Proj");
	billboardTextureID = loadDDS(texture);

	initializeBuffers();
}

void ParticleContainer::initializeBuffers()
{
	// The VBO containing the 4 vertices of the particles.
	// Thanks to instancing, they will be shared by all particles.
	static const GLfloat g_vertex_buffer_data[] = {
		-0.5f, -0.5f, 0.0f,
		0.5f, -0.5f, 0.0f,
		-0.5f, 0.5f, 0.0f,
		0.5f, 0.5f, 0.0f,
	};
	glGenBuffers(1, &billboard_vertex_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, billboard_vertex_buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

	// The VBO containing the positions and sizes of the particles
	glGenBuffers(1, &particles_position_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
	// Initialize with empty (NULL) buffer : it will be updated later, each frame.
	glBufferData(GL_ARRAY_BUFFER, max_particle_count * 4 * sizeof(GLfloat), NULL, GL_STREAM_DRAW);

	// The VBO containing the colors of the particles
	glGenBuffers(1, &particles_color_buffer);
	glBindBuffer(GL_ARRAY_BUFFER, particles_color_buffer);
	// Initialize with empty (NULL) buffer : it will be updated later, each frame.
	glBufferData(GL_ARRAY_BUFFER, max_particle_count * 4 * sizeof(GLubyte), NULL, GL_STREAM_DRAW);

}
void ParticleContainer::updateGLBuffers()
{
	glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
	glBufferData(GL_ARRAY_BUFFER, max_particle_count * 4 * sizeof(GLfloat), NULL, GL_STREAM_DRAW); // Buffer orphaning, a common way to improve streaming perf. See above link for details.
	glBufferSubData(GL_ARRAY_BUFFER, 0, render_counter * sizeof(GLfloat) * 4, g_particle_position_data);

	glBindBuffer(GL_ARRAY_BUFFER, particles_color_buffer);
	glBufferData(GL_ARRAY_BUFFER, max_particle_count * 4 * sizeof(GLubyte), NULL, GL_STREAM_DRAW); // Buffer orphaning, a common way to improve streaming perf. See above link for details.
	glBufferSubData(GL_ARRAY_BUFFER, 0, render_counter * sizeof(GLubyte) * 4, g_particle_color_data);

}
void ParticleContainer::Draw()
{
	updateGLBuffers();

	// Use our shader
	glUseProgram(programID);

	// Bind our texture in Texture Unit 0
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, billboardTextureID);
	glUniform1i(textureID, 0); //pass texureID to shader

	// sample settings for billboards
	glUniform3f(viewVec1ID, viewMat[0][0], viewMat[1][0], viewMat[2][0]);
	glUniform3f(viewVec2ID, viewMat[0][1], viewMat[1][1], viewMat[2][1]);
	glUniformMatrix4fv(vpMatID, 1, GL_FALSE, &projMat[0][0]);

	// 1st attribute buffer : vertices
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, billboard_vertex_buffer);
	glVertexAttribPointer(
		0, // attribute. No particular reason for 0, but must match the layout in the shader.
		3, // size
		GL_FLOAT, // type
		GL_FALSE, // normalized?
		0, // stride
		(void*)0 // array buffer offset
		);

	// 2nd attribute buffer : positions of particles' centers
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, particles_position_buffer);
	glVertexAttribPointer(
		1, // attribute. No particular reason for 1, but must match the layout in the shader.
		4, // size : x + y + z + size => 4
		GL_FLOAT, // type
		GL_FALSE, // normalized?
		0, // stride
		(void*)0 // array buffer offset
		);

	// 3rd attribute buffer : particles' colors
	glEnableVertexAttribArray(2);
	glBindBuffer(GL_ARRAY_BUFFER, particles_color_buffer);
	glVertexAttribPointer(
		2, // attribute. No particular reason for 1, but must match the layout in the shader.
		4, // size : r + g + b + a => 4
		GL_UNSIGNED_BYTE, // type
		GL_TRUE, // normalized? *** YES, this means that the unsigned char[4] will be accessible with a vec4 (floats) in the shader ***
		0, // stride
		(void*)0 // array buffer offset
		);

	glVertexAttribDivisor(0, 0); // particles vertices : always reuse the same 4 vertices -> 0
	glVertexAttribDivisor(1, 1); // positions : one per quad (its center) -> 1
	glVertexAttribDivisor(2, 1); // color : one per quad -> 1

	glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, render_counter);

	//disable shader
	glUseProgram(0);
}
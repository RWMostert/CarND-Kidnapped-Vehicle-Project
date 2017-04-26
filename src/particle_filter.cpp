/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 100;
    particles = vector<Particle>(num_particles);
    weights = vector<double>(num_particles);

    random_device rd;
    default_random_engine gen(rd());

    // This line creates a normal (Gaussian) distribution
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for(int i = 0; i < num_particles; i++) {
        Particle particle = Particle();
        particle.id = i;
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
        particle.weight = 1.0;

        particles[i] = particle;
        weights[i] = 1.0;
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/



    for(int i = 0; i < particles.size(); i++){

        double x_0 		= particles[i].x;
        double y_0 		= particles[i].y;
        double theta_0 	= particles[i].theta;

        double x_f 		= x_0 + (velocity/yaw_rate) * ( sin(theta_0 + yaw_rate*delta_t) - sin(theta_0) );
        double y_f 		= y_0 + (velocity/yaw_rate) * ( cos(theta_0) - cos(theta_0 + yaw_rate*delta_t) );
        double theta_f 	= theta_0 + yaw_rate * delta_t;

        // This line creates a normal (Gaussian) distribution
        random_device rd;
        default_random_engine gen(rd());

        normal_distribution<double> dist_x(x_f, std_pos[0]);
        normal_distribution<double> dist_y(y_f, std_pos[1]);
        normal_distribution<double> dist_theta(theta_f, std_pos[2]);

        particles[i].x 	= dist_x(gen);
        particles[i].y 	= dist_y(gen);
        particles[i].theta 	= dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
    //   for the fact that the map's y-axis actually points downwards.)
    //   http://planning.cs.uiuc.edu/node99.html

    // This line creates a normal (Gaussian) distribution

    for (int i = 0; i < particles.size(); i++){
        double likelihood = 1.0;

        vector<LandmarkObs> transformed_observations = vector<LandmarkObs>(observations.size());

        double x = particles[i].x;
        double y = particles[i].y;
        double theta = particles[i].theta;

        for (int j = 0; j < observations.size(); j++) {

            double obs_x = observations[j].x;
            double obs_y = observations[j].y;

            double x_trans = x + (obs_x * cos(theta)) - (obs_y * sin(theta));
            double y_trans = y + (obs_x * sin(theta)) + (obs_y * cos(theta));

            LandmarkObs transformed_observation = LandmarkObs();

            transformed_observation.id = observations[j].id;
            transformed_observation.x = x_trans;
            transformed_observation.y = y_trans;

            transformed_observations[j] = transformed_observation;
        }

        for (int j = 0; j < observations.size(); j++){
            LandmarkObs observation = transformed_observations[j];
            Map::single_landmark_s nearest_neighbor;
            double nearest_neighbor_dist = numeric_limits<double>::infinity();

            for(int k = 0; k < map_landmarks.landmark_list.size(); k++){
                Map::single_landmark_s neighbor = map_landmarks.landmark_list[k];
                double distance = sqrt(pow(neighbor.x_f - observation.x, 2) + pow(neighbor.y_f - observation.y, 2));

                if (distance < nearest_neighbor_dist) {
                    nearest_neighbor_dist = distance;
                    nearest_neighbor = neighbor;
                }
            }

            double denominator = 2 * M_PI * std_landmark[0] * std_landmark[1];
            double exp_inner = -0.5 * (pow(nearest_neighbor.x_f - observation.x, 2) / pow(std_landmark[0], 2) + pow(nearest_neighbor.y_f - observation.y, 2) / pow(std_landmark[1], 2));
            likelihood = likelihood * exp(exp_inner)/denominator;
        }
        particles[i].weight = likelihood;
    }

    // Normalise weights:
    double sum_probs = 0.0;
    for(int i = 0; i < particles.size(); i++) sum_probs = sum_probs + particles[i].weight;
    for(int i = 0; i < particles.size(); i++){
        particles[i].weight = particles[i].weight/sum_probs;
        weights[i] = particles[i].weight;
    }

}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    vector<Particle> resampled_particles = vector<Particle>(particles.size());

    random_device rd;
    default_random_engine gen(rd());
    discrete_distribution<> dist(weights.begin(), weights.end());

    for(int i = 0; i < particles.size(); i++){
        resampled_particles[i] = particles[dist(gen)];
    }

    particles = resampled_particles;
}

void ParticleFilter::write(std::string filename) {
    // You don't need to modify this file.
    std::ofstream dataFile;
    dataFile.open(filename, std::ios::app);
    for (int i = 0; i < num_particles; ++i) {
        dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
    }
    dataFile.close();
}
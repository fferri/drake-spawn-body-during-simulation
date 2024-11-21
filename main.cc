#include <iostream>
#include <random>
#include <chrono>
#include <map>

#include <Eigen/Geometry>

#include <drake/geometry/scene_graph.h>
#include <drake/geometry/drake_visualizer.h>
#include <drake/math/rigid_transform.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/plant/multibody_plant_config_functions.h>
#include <drake/systems/analysis/simulator.h>
#include <drake/systems/framework/diagram_builder.h>
#include <drake/systems/framework/leaf_system.h>
#include <drake/visualization/visualization_config_functions.h>

using namespace drake;
using namespace drake::systems;
using namespace drake::geometry;
using namespace drake::multibody;

Eigen::Quaternion<double> randomQuaternion() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    double x = dist(gen);
    double y = dist(gen);
    double z = dist(gen);
    double w = dist(gen);

    Eigen::Quaternion<double> q(w, x, y, z);
    q.normalize();  // Ensure the quaternion is valid (normalized)
    return q;
}

class Simulation {
public:
    Simulation(const MultibodyPlantConfig &cfg, std::shared_ptr<geometry::Meshcat> meshcat = {})
        : plant_config_(cfg), psg_(multibody::AddMultibodyPlant(cfg, &diagram_builder_)), meshcat_(meshcat) {}

    void copyStateFrom(const Simulation &s) {
        const MultibodyPlant<double> &plant = s.psg_.plant;
        const geometry::SceneGraph<double> &scene_graph = s.psg_.scene_graph;
        const Context<double> &ctx = s.simulator_->get_context();
        const Context<double> &pctx = plant.GetMyContextFromRoot(ctx);

        const SceneGraphInspector<double> &inspector = scene_graph.model_inspector();
        for(int i = 1; i < plant.num_bodies(); i++) {
            const auto &body = plant.get_body(BodyIndex(i));
            SpatialInertia<double> si = body.CalcSpatialInertiaInBodyFrame(pctx);
            const auto &newBody = psg_.plant.AddRigidBody(body.name(), si);
            for(GeometryId id : plant.GetCollisionGeometriesForBody(body)) {
                const geometry::Shape& geom = inspector.GetShape(id);
                const std::string &n = inspector.GetName(id);
                const ProximityProperties *pp = inspector.GetProximityProperties(id);
                psg_.plant.RegisterCollisionGeometry(newBody, math::RigidTransformd(), geom, n, *pp);
            }
            for(GeometryId id : plant.GetVisualGeometriesForBody(body)) {
                const geometry::Shape& geom = inspector.GetShape(id);
                const std::string &n = inspector.GetName(id);
                const IllustrationProperties *ip = inspector.GetIllustrationProperties(id);
                psg_.plant.RegisterVisualGeometry(newBody, math::RigidTransformd(), geom, n, *ip);
            }
            setInitialPose(newBody, body.EvalPoseInWorld(pctx));
            setInitialVelocity(newBody, body.EvalSpatialVelocityInWorld(pctx));
            if(s.isStatic(body))
                setStatic(newBody);
        }
    }

    void init() {
        MultibodyPlant<double> &plant = psg_.plant;
        plant.Finalize();

        if(meshcat_)
            visualization::AddDefaultVisualization(&diagram_builder_, meshcat_);

        diagram_ = diagram_builder_.Build();
        simulator_ = std::make_unique<Simulator<double>>(*diagram_);
        Context<double> &pctx = plant.GetMyMutableContextFromRoot(&simulator_->get_mutable_context());

        for(const auto &p : initPose_)
            if(!isStatic(*p.first))
                plant.SetFreeBodyPose(&pctx, *p.first, p.second);
        for(const auto &p : initVel_)
            if(!isStatic(*p.first))
                plant.SetFreeBodySpatialVelocity(&pctx, *p.first, p.second);

        simulator_->set_publish_every_time_step(true);
        simulator_->set_target_realtime_rate(1.0);
        simulator_->Initialize();
    }

    double getSimulationTime() {
        return simulator_->get_context().get_time();
    }

    double getSimulationTimeStep() {
        return plant_config_.time_step;
    }

    void step() {
        double t = getSimulationTime();
        double dt = getSimulationTimeStep();
        simulator_->AdvanceTo(t + dt);
    }

    const RigidBody<double>& createCuboid(const std::string &name, const double &mass, const Eigen::Vector3d &size, const Eigen::Vector4d &color = {0.5, 0.5, 0.5, 1.0}) {
        MultibodyPlant<double> &plant = psg_.plant;
        //SpatialInertia<double> inertia(mass, Eigen::Vector3d::Zero(), UnitInertia<double>::TriaxiallySymmetric((1.0 / 6.0) * mass * size(0) * size(1)));
        SpatialInertia<double> inertia = SpatialInertia<double>::SolidBoxWithMass(mass, size(0), size(1), size(2));
        const RigidBody<double> &body = plant.AddRigidBody(name, inertia);
        const geometry::Shape& shape = geometry::Box(size);
        CoulombFriction<double> f(static_friction_, dynamic_friction_);
        plant.RegisterCollisionGeometry(body, math::RigidTransformd(), shape, name + "_collision", f);
        plant.RegisterVisualGeometry(body, math::RigidTransformd(), shape, name + "_visual", color);
        return body;
    }

    void setStatic(const RigidBody<double> &body) {
        MultibodyPlant<double> &plant = psg_.plant;
        plant.WeldFrames(plant.world_frame(), body.body_frame());
        staticFlag_[&body] = true;
    }

    bool isStatic(const RigidBody<double> &body) const {
        auto it = staticFlag_.find(&body);
        return it != staticFlag_.end() && it->second == true;
    }

    void setInitialPose(const RigidBody<double> &body, const math::RigidTransform<double> &pose) {
        initPose_[&body] = pose;
    }

    void setInitialVelocity(const RigidBody<double> &body, const SpatialVelocity<double> &vel) {
        initVel_[&body] = vel;
    }

private:
    DiagramBuilder<double> diagram_builder_;
    const MultibodyPlantConfig &plant_config_;
    AddMultibodyPlantSceneGraphResult<double> psg_;
    std::unique_ptr<Diagram<double>> diagram_;
    std::unique_ptr<Simulator<double>> simulator_;
    std::shared_ptr<geometry::Meshcat> meshcat_;

    std::map<const RigidBody<double>*, math::RigidTransform<double>> initPose_;
    std::map<const RigidBody<double>*, SpatialVelocity<double>> initVel_;
    std::map<const RigidBody<double>*, bool> staticFlag_;

    const double static_friction_ = 0.8;
    const double dynamic_friction_ = 0.5; // (not used in stepped simulations)
};

int main() {
    auto meshcat = std::make_shared<geometry::Meshcat>();
    MultibodyPlantConfig cfg;
    cfg.time_step = 0.005; // s
    cfg.stiction_tolerance = 1.0E-4; // m/s
    cfg.penetration_allowance = 1.0E-4;
    cfg.discrete_contact_approximation = "sap"; // tamsi || sap || similar || lagged
    std::unique_ptr<Simulation> ps = std::make_unique<Simulation>(cfg, meshcat);
    const auto &ground = ps->createCuboid("ground", 1.0, {10, 10, 0.1}, {0.3, 0.3, 0.8, 1});
    ps->setStatic(ground);
    auto &cube = ps->createCuboid("cube", 1.0, {0.1, 0.1, 0.1}, {0.8, 0.3, 0.3, 1});
    ps->setInitialPose(cube, math::RigidTransform<double>(randomQuaternion(), Eigen::Vector3d(0, 0, 1.3)));
    ps->setInitialVelocity(cube, SpatialVelocity<double>(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 1)));
    ps->init();
    while(true) {
        double t = ps->getSimulationTime();
        double dt = ps->getSimulationTimeStep();
        if(t > 0 && fmod(t, 1) < dt) {
            std::unique_ptr<Simulation> ps1 = std::make_unique<Simulation>(cfg, meshcat);
            ps1->copyStateFrom(*ps);
            static int i = 0;
            auto &cube = ps1->createCuboid("cube-" + std::to_string(++i), 1.0, {0.1, 0.1, 0.1}, {0.8, 0.3, 0.3, 1});
            ps1->setInitialPose(cube, math::RigidTransform<double>(randomQuaternion(), Eigen::Vector3d(0, 0, 1.3)));
            ps1->setInitialVelocity(cube, SpatialVelocity<double>(Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 1)));
            ps1->init();
            ps = std::move(ps1);
        }
        ps->step();
    }
    return 0;
}

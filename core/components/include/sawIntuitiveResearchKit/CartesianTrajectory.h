#ifndef CartesianTrajectory_h
#define CartesianTrajectory_h

#include <cisstVector/vctTypes.h>
#include <utility>

class CartesianTrajectory {
public:
    static constexpr double epsilon = 1e-6;

    /** Computes point on line segment [start, start+direction] that is closest to p */
    static double closest_point(vct3 p, vct3 start, vct3 direction)
    {
        vct3 ap = p - start;
        double norm2 = direction.DotProduct(direction);
        double t = (norm2 > epsilon) ? ap.DotProduct(direction) / norm2 : 1.0;
        t = std::max(0.0, std::min(1.0, t)); // clamp to [0, 1]
        return t;
    }

    /** Computes intersection of line segment [start, start+direction] with circle
     * Return format is <is_valid, value>
     * Only allows intersection points closer to end than center is
     * */
    static std::tuple<bool, double> intersection(vct3 start, vct3 direction, vct3 center, double radius)
    {
        vct3 offset = start - center;
        double a = direction.DotProduct(direction);
        double b = 2 * direction.DotProduct(offset);
        double c = offset.DotProduct(offset) - (radius * radius);
        double discriminant = b * b - 4 * a * c;

        bool valid = discriminant >= 0.0 && a >= epsilon;
        double value = 1.0;
        if (valid) {
            value = (-b + std::sqrt(discriminant)) / (2 * a);
            value = std::max(0.0, std::min(1.0, value));
        }

        return std::tuple<bool, double>(valid, value);
    }

    /** Pure pursuit for path consisting only of line segment [start, start+direction] */
    static double pure_pursuit(vct3 start, vct3 direction, vct3 position, double look_ahead)
    {
        double proj_t = closest_point(position, start, direction);

        // intersection between look-ahead circle and straight-line path
        bool found_intersect;
        double intersect_t;
        std::tie(found_intersect, intersect_t) = intersection(start, direction, position, look_ahead);
    
        double t = found_intersect ? std::max(proj_t, intersect_t) : proj_t;
        if (t > 1.0 - epsilon) {
            return 1.0;
        }
        return t;
    }

    static vctDoubleVec feasible_velocity(vctDoubleVec max_v, vctDoubleVec max_a,
                                          vctDoubleVec current_jv, vctDoubleVec target_jv,
                                          vctDoubleVec current_jp, vctDoubleVec target_jp)
    {
        vctDoubleVec distance = target_jp - current_jp;
        vctDoubleVec jv(target_jv.size(), 0.0);
        for (size_t idx = 0; idx < jv.size(); idx++) {
            double max_delta_v = std::abs(distance[idx]) * std::sqrt(std::abs(2.0 * distance[idx] / max_a[idx]));
            jv[idx] = target_jv[idx];
            // clamp within limits imposed by velocity and acceleration limits
            jv[idx] = std::max(-max_v[idx], std::min(max_v[idx], jv[idx]));
            jv[idx] = std::max(current_jv[idx] - max_delta_v, std::min(current_jv[idx] + max_delta_v, jv[idx]));
        }

        return jv;
    }
};

#endif // CartesianTrajectory_h
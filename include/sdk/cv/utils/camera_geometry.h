/**********************************************************************************************************************
 * Copyright (c) Prophesee S.A. - All Rights Reserved                                                                 *
 *                                                                                                                    *
 * Subject to Prophesee Metavision Licensing Terms and Conditions ("License T&C's").                                  *
 * You may not use this file except in compliance with these License T&C's.                                           *
 * A copy of these License T&C's is located in the "licensing" folder accompanying this file.                         *
 **********************************************************************************************************************/

#ifndef METAVISION_SDK_CV_CAMERA_GEOMETRY_H
#define METAVISION_SDK_CV_CAMERA_GEOMETRY_H

#include "metavision/sdk/cv/utils/camera_geometry_base.h"

namespace Metavision {

/// @brief Camera geometry's implementation
/// @tparam Projection The projection model used to map points
template<typename Projection>
class CameraGeometry : public CameraGeometryBase<typename Projection::underlying_type> {
public:
    using Base         = CameraGeometryBase<typename Projection::underlying_type>;
    using Vec2         = typename Base::Vec2;
    using Vec2i        = typename Base::Vec2i;
    using Vec3         = typename Base::Vec3;
    using Vec2Ref      = typename Base::Vec2Ref;
    using Vec2RefConst = typename Base::Vec2RefConst;
    using Vec3RefConst = typename Base::Vec3RefConst;
    using Mat2CM       = typename Base::Mat2CM;
    using Mat2RMRef    = typename Base::Mat2RMRef;
    using Mat2CMRef    = typename Base::Mat2CMRef;
    using Mat3RMRef    = typename Base::Mat3RMRef;
    using Mat3CMRef    = typename Base::Mat3CMRef;
    using T            = typename Projection::underlying_type;

    CameraGeometry(const Projection &projection) : projection_(projection) {}

    virtual ~CameraGeometry() = default;

    const Projection &get_projection() const {
        return projection_;
    }

    const Vec2i &get_image_size() const final {
        return projection_.get_image_size();
    }

    T get_distance_to_image_plane() const final {
        return projection_.get_distance_to_image_plane();
    }

    std::unique_ptr<CameraGeometryBase<T>> clone() const final {
        return std::make_unique<CameraGeometry<Projection>>(*this);
    }

    void get_undist_norm_to_undist_img_transform(Mat3RMRef m) const final {
        projection_.get_undist_norm_to_undist_img_transform(m);
    }

    void get_undist_norm_to_undist_img_transform(Mat3CMRef m) const final {
        projection_.get_undist_norm_to_undist_img_transform(m);
    }

    void camera_to_img(Vec3RefConst pt_c, Vec2Ref pt_dist_img) const final {
        projection_.camera_to_img(pt_c, pt_dist_img);
    }

    void camera_to_undist_img(Vec3RefConst pt_c, Vec2Ref pt_dist_img) const final {
        projection_.camera_to_undist_img(pt_c, pt_dist_img);
    }

    void undist_norm_to_undist_img(Vec2RefConst pt_undist_norm, Vec2Ref pt_undist_img) const final {
        projection_.undist_norm_to_undist_img(pt_undist_norm, pt_undist_img);
    }

    void undist_norm_to_dist_norm(Vec2RefConst pt_undist_norm, Vec2Ref pt_dist_norm) const final {
        projection_.undist_norm_to_dist_norm(pt_undist_norm, pt_dist_norm);
    }

    void undist_img_to_undist_norm(Vec2RefConst pt_undist_img, Vec2Ref pt_undist_norm) const final {
        projection_.undist_img_to_undist_norm(pt_undist_img, pt_undist_norm);
    }

    void undist_norm_to_img(Vec2RefConst pt_undist_norm, Vec2Ref pt_dist_img) const final {
        projection_.undist_norm_to_img(pt_undist_norm, pt_dist_img);
    }

    void img_to_undist_norm(Vec2RefConst pt_dist_img, Vec2Ref pt_undist_norm) const final {
        projection_.img_to_undist_norm(pt_dist_img, pt_undist_norm);
    }

    void vector_undist_norm_to_img(Vec2RefConst ctr_undist_norm, Vec2RefConst vec_undist_norm, Vec2Ref ctr_dist_img,
                                   Vec2Ref vec_dist_img) const final {
        Mat2CM Jdist;
        projection_.get_undist_norm_to_img_jacobian(ctr_undist_norm, ctr_dist_img, Jdist);

        vec_dist_img = Jdist * vec_undist_norm;
        vec_dist_img.normalize();
    }

    void vector_img_to_undist_norm(Vec2RefConst ctr_dist_img, Vec2RefConst vec_dist_img, Vec2Ref ctr_undist_norm,
                                   Vec2Ref vec_undist_norm) const final {
        Mat2CM Jundist;
        projection_.get_img_to_undist_norm_jacobian(ctr_dist_img, ctr_undist_norm, Jundist);

        vec_undist_norm = Jundist * vec_dist_img;
        vec_undist_norm.normalize();
    }

    void get_undist_norm_to_img_jacobian(Vec2RefConst pt_undist_norm, Vec2Ref pt_dist_img, Mat2RMRef J) const final {
        projection_.get_undist_norm_to_img_jacobian(pt_undist_norm, pt_dist_img, J);
    }

    void get_undist_norm_to_img_jacobian(Vec2RefConst pt_undist_norm, Vec2Ref pt_dist_img, Mat2CMRef J) const final {
        projection_.get_undist_norm_to_img_jacobian(pt_undist_norm, pt_dist_img, J);
    }

    void get_img_to_undist_norm_jacobian(Vec2RefConst pt_dist_img, Vec2Ref pt_undist_norm, Mat2RMRef J) const final {
        projection_.get_img_to_undist_norm_jacobian(pt_dist_img, pt_undist_norm, J);
    }

    void get_img_to_undist_norm_jacobian(Vec2RefConst pt_dist_img, Vec2Ref pt_undist_norm, Mat2CMRef J) const final {
        projection_.get_img_to_undist_norm_jacobian(pt_dist_img, pt_undist_norm, J);
    }

private:
    Projection projection_;
};

} // namespace Metavision

#endif // METAVISION_SDK_CV_CAMERA_GEOMETRY_H

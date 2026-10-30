import React from "react";
import { MultimodalScore } from "@/app/types/patient";

interface MultimodalRiskScoreProps {
  multimodalScore?: MultimodalScore | null;
}

export default function MultimodalRiskScore({
  multimodalScore,
}: MultimodalRiskScoreProps) {
  if (!multimodalScore) {
    return null;
  }

  const {
    score,
    clinical,
    byra,
    behsof,
    agreement,
  } = multimodalScore;

  const scoreStyle = getScoreStyle(score);
  const agreementStyle = getAgreementStyle(agreement);

  return (
    <div className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm">

      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-50">
            <svg
              className="h-5 w-5 text-blue-600"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 12c0 5.591 3.824 10.291 9 11.622C17.176 22.291 21 17.591 21 12c0-1.268-.196-2.49-.56-3.64"
              />
            </svg>
          </div>

          <div>
            <h2 className="text-xl font-semibold text-gray-900">
              Multimodal AI Risk Score
            </h2>

            <p className="mt-1 text-sm text-gray-500">
              Combined evidence from clinical and ultrasound AI models
            </p>
          </div>
        </div>
      </div>

      {/* Main Score */}
      <div className="mb-8 flex flex-col items-center justify-center">

        <div
          className={`text-5xl font-bold ${scoreStyle.text}`}
        >
          {score.toFixed(1)}
        </div>

        <div className="mt-1 text-sm text-gray-500">
          out of 100
        </div>

        <div
          className={`mt-3 rounded-full px-4 py-1.5 text-sm font-medium ${agreementStyle.badge}`}
        >
          Evidence: {agreement}
        </div>
      </div>

      {/* Model Contributions */}
      <div className="space-y-5">

        {/* Clinical */}
        <ModelRow
          name="Clinical ML"
          probability={clinical.probability}
          weight={clinical.weight}
          contribution={clinical.contribution}
          evidenceLevel={clinical.evidence_level}
        />

        {/* Byra */}
        <ModelRow
          name="Byra ViT"
          probability={byra.probability}
          weight={byra.weight}
          contribution={byra.contribution}
          evidenceLevel={byra.evidence_level}
        />

        {/* BEHSOF */}
        <ModelRow
          name="BEHSOF ViT"
          probability={behsof.probability}
          weight={behsof.weight}
          contribution={behsof.contribution}
          evidenceLevel={behsof.evidence_level}
        />

      </div>

      {/* Disclaimer */}
      <div className="mt-6 rounded-lg border border-gray-100 bg-gray-50 p-3 text-xs leading-relaxed text-gray-500">
        This score represents an AI-based aggregation of model outputs
        and is not a clinically validated disease probability.
      </div>

    </div>
  );
}


/* -------------------------------------------------
   Model Row
------------------------------------------------- */

interface ModelRowProps {
  name: string;
  probability: number;
  weight: number;
  contribution: number;
  evidenceLevel: string;
}

function ModelRow({
  name,
  probability,
  weight,
  contribution,
  evidenceLevel,
}: ModelRowProps) {

  const style = getEvidenceStyle(evidenceLevel);

  return (
    <div>

      {/* Title */}
      <div className="mb-2 flex items-center justify-between">

        <div>
          <div className="font-medium text-gray-900">
            {name}
          </div>

          <div className="mt-0.5 flex items-center gap-1.5 text-xs">

            <span
              className={`h-2 w-2 rounded-full ${style.dot}`}
            />

            <span className={style.text}>
              Evidence: {evidenceLevel}
            </span>

          </div>
        </div>

        <div className="text-right">

          <div className="font-semibold text-gray-900">
            {probability.toFixed(1)}%
          </div>

          <div className="text-xs text-gray-500">
            Weight {(weight * 100).toFixed(0)}%
          </div>

        </div>

      </div>

      {/* Progress */}
      <div className="h-2 overflow-hidden rounded-full bg-gray-100">

        <div
          className={`h-full rounded-full transition-all duration-500 ${style.progress}`}
          style={{
            width: `${Math.min(Math.max(probability, 0), 100)}%`,
          }}
        />

      </div>

      {/* Contribution */}
      <div className="mt-1 text-right text-xs text-gray-500">
        Contribution:{" "}
        <span className="font-medium text-gray-700">
          {contribution.toFixed(2)}
        </span>
      </div>

    </div>
  );
}


/* -------------------------------------------------
   Evidence Colors
------------------------------------------------- */

function getEvidenceStyle(level: string) {

  switch (level.toLowerCase()) {

    case "high":
      return {
        text: "text-red-600",
        dot: "bg-red-500",
        progress: "bg-red-500",
      };

    case "moderate":
      return {
        text: "text-orange-600",
        dot: "bg-orange-500",
        progress: "bg-orange-500",
      };

    case "low":
    default:
      return {
        text: "text-green-600",
        dot: "bg-green-500",
        progress: "bg-green-500",
      };
  }
}


/* -------------------------------------------------
   Score Colors
------------------------------------------------- */

function getScoreStyle(score: number) {

  if (score >= 70) {
    return {
      text: "text-red-600",
    };
  }

  if (score >= 40) {
    return {
      text: "text-orange-500",
    };
  }

  return {
    text: "text-green-600",
  };
}


/* -------------------------------------------------
   Agreement Colors
------------------------------------------------- */

function getAgreementStyle(agreement: string) {

  switch (agreement.toLowerCase()) {

    case "strong":
      return {
        badge: "bg-red-50 text-red-700",
      };

    case "moderate":
      return {
        badge: "bg-orange-50 text-orange-700",
      };

    case "mixed":
    default:
      return {
        badge: "bg-blue-50 text-blue-700",
      };
  }
}
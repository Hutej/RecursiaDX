import React from "react"
import { cn } from "@/lib/utils"

export function ImageCard({
  src,
  alt,
  className,
  ...props
}) {
  const handleLoad = () => {
    console.log('✅ Image loaded successfully:', src)
  }

  const handleError = () => {
    console.error('❌ Image failed to load:', src)
  }

  return (
    <img
      src={src}
      alt={alt}
      className={cn("rounded-lg border", className)}
      onLoad={handleLoad}
      onError={handleError}
      {...props}
    />
  )
}
